// SeqGen.cu - GPU LZ77 sequence generator for the 7-Zip zstd codec.
//
// Chunk-granular G-Match-style pipeline.  Each chunk is an independent
// unit submitted on its own CUDA stream by the host ring: the device
// never waits for a wave boundary and H2D/D2H transfers of one chunk
// hide under another chunk's compute.
//
//   kernel 1 (zgpu_cand_kernel): one thread PER INPUT POSITION
//     finds that position's best match candidate: hash-table probe
//     (4 slots/bucket, shared per chunk), best-of-4 capped extension,
//     plus direct short-offset probes (d=1,2,4,8,16,32) which cover
//     the repcode-style matches that a hash table cannot see across
//     lanes.  Result per position: packed u32 (off<<6 | len<=63).
//     Threads are fully independent -> ~1M threads/wave, real
//     occupancy, latency hidden by sheer parallelism.
//     The hash table is NEVER cleared: stale entries from a previous
//     chunk are validated by byte comparison anyway, so the worst they
//     can do is waste a probe - output stays correct.
//
//   kernel 2 (zgpu_pick_kernel): one warp per segment replays the
//     sequential parse over the precomputed candidates.  Lane k
//     loads the candidate for position base+k; lane 0 walks the
//     window greedily with lazy matching (pos+1 comparison is one
//     shfl).  Matches that hit the 63-byte cap are re-extended by
//     the whole warp (8-byte slice per lane + ballot).
//
//   kernel 3 (zgpu_pack_kernel): compacts the per-segment sequence
//     lists into one contiguous range; destination offsets are a
//     tiny in-kernel prefix over seqCounts (segs is small), so the
//     host only needs the packed byte count for the final D2H.
//
// Emitted records match ZSTD_Sequence layout {offset, litLength,
// matchLength, rep=0}; the host stitches per-segment outputs into
// one sequence list per chunk for ZSTD_compressSequences().

#include <cuda_runtime.h>
#include <stdint.h>

#include <string.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;
typedef unsigned char u8;

#define ZGPU_MINMATCH 4
#define ZGPU_EXTCAP   63
#ifndef ZGPU_LENCAP
#define ZGPU_LENCAP   128   /* stored-len field is 8 bits; a stored
                             value of LENCAP means "verified
                             >=LENCAP" and consumers re-extend -
                             everything below is the exact verified
                             match length */
#endif
#define ZGPU_FULLMASK 0xFFFFFFFFu

struct ZSeq { u32 off, lit, mlen, rep; };

static __device__ __forceinline__ u32 rd32(const u8 *p)
{
  u32 v; memcpy(&v, p, 4); return v;
}
static __device__ __forceinline__ u64 rd64(const u8 *p)
{
  u64 v; memcpy(&v, p, 8); return v;
}

/* common prefix length of a[..]/b[..] bounded by limit, via 8-byte XOR.
   Stride-16 main loop: the two rd64 pairs are independent so they
   issue in ONE latency step; the old 8B chain paid one memory
   latency per 8 bytes, which dominated on long matches.  (Stride-32
   measured worse: too much wasted bandwidth on short matches.) */
static __device__ __forceinline__ u32 matchlen(const u8 *a, const u8 *b,
    u32 limit)
{
  u32 l = 0;
  while (l + 16 <= limit) {
    u64 x0 = rd64(a + l)     ^ rd64(b + l);
    u64 x1 = rd64(a + l + 8) ^ rd64(b + l + 8);
    if (x0 | x1) {
      u64 x = x0 ? x0 : x1;
      l += x0 ? 0 : 8;
      return l + (u32)((__ffsll((long long)x) - 1) >> 3);
    }
    l += 16;
  }
  while (l + 8 <= limit) {
    u64 x = rd64(a + l) ^ rd64(b + l);
    if (x == 0) { l += 8; continue; }
    return l + (u32)((__ffsll((long long)x) - 1) >> 3);
  }
  while (l < limit && a[l] == b[l]) l++;
  return l;
}

static __device__ __forceinline__ u32 matchlen_cap(const u8 *a,
    const u8 *b, u32 limit)
{
  return matchlen(a, b, limit < ZGPU_EXTCAP ? limit : ZGPU_EXTCAP);
}

/* candidate evaluation caps at EXTCAP for speed; when the winning
   candidate hit the cap, keep extending once to the storage cap so
   the packed len is exact for anything below LENCAP.  Moves the
   serial re-extension out of the pick kernel's critical path into
   this position-parallel kernel. */
static __device__ __forceinline__ u32 zgpu_truelen(const u8 *s,
    u32 p, u32 off, u32 len, u32 hi)
{
  if (len >= ZGPU_EXTCAP && off && p + len < hi) {
    u32 lim = hi - p - len;
    if (lim > ZGPU_LENCAP - ZGPU_EXTCAP)
      lim = ZGPU_LENCAP - ZGPU_EXTCAP;
    len += matchlen(s + p + len, s + p + len - off, lim);
  }
  return len;
}

/*
 * Single-load candidate evaluation: one 8-byte XOR does the 4-byte
 * minmatch verify (low word) AND yields the next 4 bytes of match
 * length (high word, located via ffs).  Only when all 8 bytes match
 * do we pay for a real extension loop (capped at EXTCAP-8 more).
 * Requires p + 8 <= hi (caller guarantees).
 */
static __device__ __forceinline__ u32 candLen64(const u8 *s, u32 p,
    u32 c, u64 pv, u32 hi)
{
  u64 x = rd64(s + c) ^ pv;
  if ((u32)x)
    return 0;
  u32 h32 = (u32)(x >> 32);
  if (h32)
    return 4 + ((u32)(__ffs((int)h32) - 1) >> 3);
  u32 lim = hi - p - 8;
  u32 cap = lim < (ZGPU_EXTCAP - 8) ? lim : (ZGPU_EXTCAP - 8);
  return 8 + matchlen(s + p + 8, s + c + 8, cap);
}

/* warp-cooperative full extension: len0 bytes already verified;
   each lane covers an 8-byte slice per round (256 B), ballot finds
   the first failing lane, lane 0 resolves it byte-wise (<=8 iters).
   All lanes must call together; the result is broadcast to all. */
static __device__ __forceinline__ u32 warp_extend(const u8 *s, u32 pos,
    u32 off, u32 len0, u32 hi, u32 lane)
{
  u32 len = len0;
  for (;;) {
    u32 b = pos + len + lane * 8;
    int ok = (b + 8 <= hi) &&
             (rd64(s + b) == rd64(s + b - off));
    unsigned bal = __ballot_sync(ZGPU_FULLMASK, ok);
    if (bal == ZGPU_FULLMASK) { len += 256; continue; }
    unsigned fb = __ffs(~bal) - 1;
    len += fb * 8;
    if (lane == 0) {
      u32 lim = 0;
      if (pos + len < hi) {
        lim = hi - pos - len;
        if (lim > 8) lim = 8;
      }
      if (lim)
        len += matchlen(s + pos + len, s + pos - off + len, lim);
    }
    return __shfl_sync(ZGPU_FULLMASK, len, 0);
  }
}

/* ---------------- kernel 1: candidate generation ------------------
 * One thread per position of ONE chunk (s = chunk base, n bytes).
 * Segment bounds matter because the pick kernel is segment-parallel:
 * candidates must never point outside their own segment's start so a
 * sequence always lies inside one segment.
 */
template <bool KMODE>
__global__ void __launch_bounds__(512, 4) zgpu_cand_kernel(
    const u8 *__restrict__ s,
    u32 n,
    u32 segsPerChunk,
    u32 hashLog,
    u32 *__restrict__ hashTab,
    u32 *__restrict__ candOut,
    u32 *__restrict__ candK)
{
  u32 p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;

  u32 segLen = (n + segsPerChunk - 1) / segsPerChunk;
  u32 si = p / segLen;
  u32 lo = si * segLen;
  u32 hi = min(n, lo + segLen);

  u32 out = 0;                                  /* len=0: no cand */
  /* top-4 alternative candidates for the DP parse kernel: each
     position is owned by exactly one thread so the row needs no
     atomics - just a sorted 4-entry insert scored like the primary
     pick (len*8 - offsetbits).  Every path below writes the whole
     16B row exactly once (zeros when empty): stale rows from the
     previous chunk on this buffer must never survive. */
  u32 k4[4] = { 0, 0, 0, 0 };
  u32 ks[4] = { 0, 0, 0, 0 };
  /* KMODE is a compile-time flag: on the pick path all top-4
     tracking code vanishes entirely (saves ~8 live registers) */
#define ZGPU_FLUSHK4() do { if (KMODE) \
    *(uint4 *)(candK + (size_t)p * 4) = \
        make_uint4(k4[0], k4[1], k4[2], k4[3]); } while (0)
#define ZGPU_INS(len_, off_, sc_) do { \
        u32 pk_ = (((len_) < ZGPU_LENCAP ? (len_) : ZGPU_LENCAP) \
            << 24) | (off_); \
        if (KMODE && pk_ != k4[0] && pk_ != k4[1] && \
            pk_ != k4[2] && pk_ != k4[3]) { \
          _Pragma("unroll") \
          for (int q_ = 0; q_ < 4; q_++) { \
            if ((sc_) > ks[q_] || \
                ((sc_) == ks[q_] && pk_ < k4[q_])) { \
              _Pragma("unroll") \
              for (int r_ = 3; r_ > q_; r_--) { \
                ks[r_] = ks[r_-1]; k4[r_] = k4[r_-1]; } \
              ks[q_] = (sc_); k4[q_] = pk_; break; } } } \
      } while (0)
  /* neighbour-offer early-out: a match found at p-1/p-2 seeds a
     strong candidate here via atomicMax.  If one already landed,
     the position is almost certainly inside a long match - skip
     the whole probe pipeline.  Order-dependent, but every outcome
     is a valid candidate so worst case is a slightly different
     (still correct) choice on racy positions. */
  u32 have = candOut[p] >> 24;
  if (p + ZGPU_MINMATCH <= hi) {
    u32 myOff = 0, myLen = 0;
    u32 h, v = 0;
    u64 pv = 0;
    if (p + 8 <= hi) {
      pv = rd64(s + p);
      /* byte-run fast path: a uniform 8-byte window's best match is
         trivially off=1; also spares the hash bucket a pile-up of
         identical keys */
      if (pv == 0x0101010101010101ULL * (u8)pv && p >= 1) {
        myLen = candLen64(s, p, p - 1, pv, hi);
        if (myLen) myOff = 1;
        myLen = zgpu_truelen(s, p, myOff, myLen, hi);
        out = myLen ? (((myLen < ZGPU_LENCAP ? myLen : ZGPU_LENCAP)
            << 24) | myOff) : 0;
        if (out) atomicMax(&candOut[p], out);
        k4[0] = out;
        ZGPU_FLUSHK4();
        return;
      }
    }
    /* slot packing: (tag10 << 22) | pos22.  The tag is a 10-bit
       fingerprint taken just BELOW the bucket-index bits, so a
       mismatching slot is rejected before its rd64 verify load -
       saves ~7 dependent loads per position.  Chunk n must stay
       below 1<<22. */
    u32 tag1, tag2;
    if (p + 8 <= hi) {
      /* 64-bit avalanche hash: better spread than the 4-byte
         multiply-shift, so buckets hold more diverse positions */
      pv = rd64(s + p);
      u64 m1 = pv * 0x9E3779B185EBCA87ULL;
      h = (u32)(m1 >> (64 - hashLog));
      tag1 = (u32)(m1 >> (64 - hashLog - 10)) & 0x3FFu;
      u64 m2 = pv * 0xC2B2AE3D27D4EB4ULL;
      tag2 = (u32)(m2 >> (64 - hashLog - 10)) & 0x3FFu;
    } else {
      u32 m = v = rd32(s + p);
      m *= 2654435761u;
      h = m >> (32 - hashLog);
      tag1 = tag2 = (m >> (32 - hashLog - 10)) & 0x3FFu;
    }
    u32 *b = hashTab + 4 * h;
    u32 *b2 = NULL, *b3 = NULL, *b4 = NULL, *b5 = NULL;
    u32 tag3 = 0, tag4 = 0, tag5 = 0;
    if (pv) {
      /* second decorrelated probe of the same table doubles the
         effective bucket depth; the slot write stays unconditional
         (cheap, keeps h2 dense) while the bucket LOAD is gated on a
         weak first probe - unconditional loads cost ~15% wall time */
      u32 h2 =
          (u32)((pv * 0xC2B2AE3D27D4EB4ULL) >> (64 - hashLog));
      b2 = hashTab + 4 * h2;
      /* third probe: only LOADED when both probes stay weak - the
         extra bucket costs ~16B DRAM only on hard positions */
      u32 h3 =
          (u32)((pv * 0xD6E8FEB86659FD93ULL) >> (64 - hashLog));
      tag3 =
          (u32)((pv * 0xD6E8FEB86659FD93ULL) >> (64 - hashLog - 10))
              & 0x3FFu;
      b3 = hashTab + 4 * h3;
      tag4 =
          (u32)((pv * 0xA24BAED4963EE407ULL) >> (64 - hashLog - 10))
              & 0x3FFu;
      b4 = hashTab + 4 *
          (u32)((pv * 0xA24BAED4963EE407ULL) >> (64 - hashLog));
      tag5 =
          (u32)((pv * 0x9FB21C651E98DF25ULL) >> (64 - hashLog - 10))
              & 0x3FFu;
      b5 = hashTab + 4 *
          (u32)((pv * 0x9FB21C651E98DF25ULL) >> (64 - hashLog));
    }
    if (have >= 20) {
      /* covered by a strong neighbour offer: skip the bucket load
         and every eval, but STILL store both slots AND set the
         prefix bit - covered positions are prime match sources
         for later positions sharing this prefix */
      b[p & 3] = (tag1 << 22) | p;
      if (b2) b2[(p + 2) & 3] = (tag2 << 22) | p;
      if (b3) b3[(p + 1) & 3] = (tag3 << 22) | p;
      if (b4) b4[(p + 3) & 3] = (tag4 << 22) | p;
      if (b5) b5[(p) & 3] = (tag5 << 22) | p;
      /* relay coverage forward: the offer at p is itself a match of
         len `have` at `off`, so p+d holds a len-(have-d) suffix.
         Without this hop the early-out chain dies after 4 covered
         positions and the rest of a long match re-evals anyway */
      u32 ent = candOut[p];          /* freshest: offers may have
                                        landed since the top read */
      u32 off = ent & 0xFFFFFFu;
      u32 hlen = ent >> 24;
      for (u32 d = 1; d <= 4 && hlen > d + 19; d++) {
        if (p + d >= hi) break;
        /* sentinel propagation: a capped (>=LENCAP) parent keeps
           the cap so the covered position re-extends for truth.
           Read-check before the atomic: atomicMax is monotone, so
           an existing value >= the offer makes the write a no-op -
           skipping it spares atomic serialization on hot runs */
        u32 ol = (hlen >= ZGPU_LENCAP) ? ZGPU_LENCAP : hlen - d;
        u32 pk = (ol << 24) | off;
        if (candOut[p + d] < pk) atomicMax(&candOut[p + d], pk);
      }
      ZGPU_FLUSHK4();
      return;
    }
    const uint4 bq = *(const uint4 *)b;   /* one 16B load, not 4 -
                       MUST precede our own store below or we evict
                       a candidate before reading it */
    b[p & 3] = (tag1 << 22) | p;  /* random-slot replace: 1 write vs
                       4-way shift storms; bucket still keeps 4-deep
                       diversity */
    if (b2) b2[(p + 2) & 3] = (tag2 << 22) | p;

    if (p + 8 <= hi) {
      /* fast path: one rd64 per candidate does verify + extension.
         Tie-break toward the SMALLER offset: equal-length matches
         are not equal once the offset code is priced. */
      u32 l;
      /* price model: a match is worth roughly len*8 bits of saved
         literals minus ~log2(off) bits of offset code.  Score =
         (len<<3) - highbit(off) - picks len8@off64 over len8@off1M
         AND len5@off8 over len4@off512K. */
      u32 myScore = 0;
#define ZGPU_EVAL(svv, tagv) do { \
        u32 sv_ = (svv); \
        u32 cc_ = sv_ & 0x3FFFFFu; \
        if ((sv_ >> 22) == (tagv) && cc_ < p) { \
          u32 ll_ = candLen64(s, p, cc_, pv, hi); \
          if (ll_) { \
            u32 oo_ = p - cc_; \
            u32 sc_ = (ll_ << 3) - (31 - __clz(oo_)); \
            ZGPU_INS(ll_, oo_, sc_); \
            if ((int)(sc_ - myScore) > 0 || \
                (sc_ == myScore && oo_ < myOff)) \
              { myLen = ll_; myOff = oo_; myScore = sc_; } \
          } \
        } } while (0)
      ZGPU_EVAL(bq.x, tag1);
      ZGPU_EVAL(bq.y, tag1);
      /* probe2's first two slots while the primary verdict is still
         weak: measured after c0/c1 only, so deep primary hits do
         not rob probe2 of its chance */
      uint4 bq2 = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu,
          0xFFFFFFFFu, 0xFFFFFFFFu);
      if (myLen < 32 && b2) {
        bq2 = *(const uint4 *)b2;
        ZGPU_EVAL(bq2.x, tag2);
        ZGPU_EVAL(bq2.y, tag2);
      }

      /* direct short-offset probes: recover repcode-style matches
         at offsets 1..256 that hash ordering cannot express.
         Backward reach may legally cross the segment edge - only
         the forward extent is capped at hi.  All probes are <2048
         so they feed the repcode-friendly track as well. */
#define ZGPU_SPROBE(d_) do { \
        if (p >= (d_) && \
            (l = candLen64(s, p, p - (d_), pv, hi))) { \
          ZGPU_INS(l, (u32)(d_), (l << 3) - (31 - __clz((u32)(d_)))); \
          if (l > myLen) { myLen = l; myOff = (d_); } \
        } } while (0)
      if (myLen == 0) {
        /* window probes for d<=8: the comparison window [p-d, p-d+8)
           lies entirely inside [p-8, p+8) - build it with shifts
           from ONE trailing load instead of a separate rd64 per
           probe.  cd_ reconstructs rd64(p-d) via:
             high d bytes of aw | low (8-d) bytes of pv          */
        if (p >= 8) {
          u64 aw = rd64(s + p - 8);
#define ZGPU_WPROBE(d_) do { \
            u64 cd_ = ((d_) == 8) ? aw \
                : ((aw >> ((8 - (d_)) * 8)) | (pv << ((d_) * 8))); \
            u64 x_ = cd_ ^ pv; \
            if (!(u32)x_) { \
              u32 h32_ = (u32)(x_ >> 32); \
              u32 ll_; \
              if (h32_) \
                ll_ = 4 + ((u32)(__ffs((int)h32_) - 1) >> 3); \
              else { \
                u32 lim_ = hi - p - 8; \
                u32 cap_ = lim_ < (ZGPU_EXTCAP - 8) \
                    ? lim_ : (ZGPU_EXTCAP - 8); \
                ll_ = 8 + matchlen(s + p + 8, s + p + 8 - (d_), \
                    cap_); \
              } \
              ZGPU_INS(ll_, (u32)(d_), \
                  (ll_ << 3) - (31 - __clz((u32)(d_)))); \
              if (ll_ > myLen) { myLen = ll_; myOff = (d_); } \
            } } while (0)
          ZGPU_WPROBE(1);
          ZGPU_WPROBE(2);
          ZGPU_WPROBE(3);
          ZGPU_WPROBE(4);
          ZGPU_WPROBE(5);
          ZGPU_WPROBE(6);
          ZGPU_WPROBE(7);
          ZGPU_WPROBE(8);
#undef ZGPU_WPROBE
        }
        ZGPU_SPROBE(12);
        ZGPU_SPROBE(16);
        ZGPU_SPROBE(24);
        ZGPU_SPROBE(32);
        ZGPU_SPROBE(64);
        ZGPU_SPROBE(128);
        ZGPU_SPROBE(256);
      }

      /* deep primary slots evaluated unconditionally: a deeper
         hit often outlives the shallow one even when myLen>=8 */
      ZGPU_EVAL(bq.z, tag1);
      ZGPU_EVAL(bq.w, tag1);
      if (myLen < 32) {
        ZGPU_EVAL(bq2.z, tag2);
        ZGPU_EVAL(bq2.w, tag2);
      }
      /* weak-everywhere positions get one more bucket's depth: the
         load is skipped on any len>=16 verdict so hot regions never
         pay for it */
      if (myLen < 24 && b3) {
        const uint4 bq3 = *(const uint4 *)b3;
        ZGPU_EVAL(bq3.x, tag3);
        ZGPU_EVAL(bq3.y, tag3);
        ZGPU_EVAL(bq3.z, tag3);
        ZGPU_EVAL(bq3.w, tag3);
      }
      /* 4th bucket only when everything above failed outright -
         rare on structured data but common on high-entropy text */
      if (myLen < 16 && b4) {
        const uint4 bq4 = *(const uint4 *)b4;
        ZGPU_EVAL(bq4.x, tag4);
        ZGPU_EVAL(bq4.y, tag4);
        ZGPU_EVAL(bq4.z, tag4);
        ZGPU_EVAL(bq4.w, tag4);
      }
      /* 5th bucket: last resort for positions that found nothing
         at all - the deepest "chain hop" approximation */
      if (!myLen && b5) {
        const uint4 bq5 = *(const uint4 *)b5;
        ZGPU_EVAL(bq5.x, tag5);
        ZGPU_EVAL(bq5.y, tag5);
        ZGPU_EVAL(bq5.z, tag5);
        ZGPU_EVAL(bq5.w, tag5);
      }
      /* deferred stores: written after the loads so our own slot
         can never evict a candidate before it is read */
      if (b3) b3[(p + 1) & 3] = (tag3 << 22) | p;
      if (b4) b4[(p + 3) & 3] = (tag4 << 22) | p;
      if (b5) b5[(p) & 3] = (tag5 << 22) | p;
#undef ZGPU_EVAL
    } else {
      /* tail (<=4B left in segment): scalar verify + bounded extend.
         Slots are tag-packed by 8-byte writers using a different
         hash family, so we decode pos only and let rd32 verify. */
      u32 lim = hi - p - ZGPU_MINMATCH;
#define ZGPU_TEVAL(svv) do { \
        u32 cc_ = (svv) & 0x3FFFFFu; \
        if (cc_ < p && rd32(s + cc_) == v) { \
          u32 l = ZGPU_MINMATCH + \
              matchlen_cap(s + p + 4, s + cc_ + 4, lim); \
          u32 oo_ = p - cc_; \
          ZGPU_INS(l, oo_, (l << 3) - (31 - __clz(oo_))); \
          if (l > myLen) { myLen = l; myOff = oo_; } \
        } } while (0)
      ZGPU_TEVAL(bq.x);
      ZGPU_TEVAL(bq.y);
      ZGPU_TEVAL(bq.z);
      ZGPU_TEVAL(bq.w);
#undef ZGPU_TEVAL
      if (myLen == 0) {
#define ZGPU_TPROBE(d_) do { \
        if (p >= (d_) && rd32(s + p - (d_)) == v) { \
          u32 l = ZGPU_MINMATCH + \
              matchlen_cap(s + p + 4, s + p - (d_) + 4, lim); \
          ZGPU_INS(l, (u32)(d_), \
              (l << 3) - (31 - __clz((u32)(d_)))); \
          if (l > myLen) { myLen = l; myOff = (d_); } \
        } } while (0)
        ZGPU_TPROBE(1);
        ZGPU_TPROBE(2);
        ZGPU_TPROBE(4);
        ZGPU_TPROBE(8);
        ZGPU_TPROBE(16);
        ZGPU_TPROBE(32);
#undef ZGPU_TPROBE
      }
    }
    if (myLen) {
      /* resolve the winner to its true length (<=LENCAP) so the
         pick kernel trusts the stored len without re-extending */
      if (myLen >= ZGPU_EXTCAP)
        myLen = zgpu_truelen(s, p, myOff, myLen, hi);
      /* len-major pack (len<<24 | off): lets atomicMax merge
         "backward-extension offers" below with plain writes - the
         largest length always wins, offset breaks ties (higher off
         wins the max but the len term dominates) */
      out = ((myLen < ZGPU_LENCAP ? myLen : ZGPU_LENCAP) << 24)
          | myOff;
      /* backward-extension offers: if this match also holds 1..3
         bytes earlier, seed cand[p-d] - optimal starts often sit
         just BEFORE the detection point and greedy can't see them */
      if (myLen >= 8) {
        for (u32 d = 1; d <= 5 && myLen + d <= ZGPU_LENCAP; d++) {
          if (p < lo + d || myOff + d > p)
            break;
          if (s[p - d] != s[p - d - myOff])
            break;
          u32 pk = ((myLen + d) << 24) | myOff;
          if (candOut[p - d] < pk)
            atomicMax(&candOut[p - d], pk);
        }
        /* forward-continuation offers: the tail of this match is
           unconditionally a shorter match at p+1..p+4 - no byte
           check needed.  Restricted to strong matches: weak big-
           offset offers would evict better own-candidates (the
           atomicMax merge is len-major, price-blind).  Reaching
           p+4 also lets the offer-early-out skip the probe work
           inside long matches entirely. */
        if (myLen >= 16) {
          unsigned lim = myLen >= 36 ? 4 : 2;
          for (u32 d = 1; d <= lim && myLen > d; d++) {
            if (p + d >= hi) break;
            u32 ol = (myLen >= ZGPU_LENCAP) ? ZGPU_LENCAP
                                          : myLen - d;
            u32 pk = (ol << 24) | myOff;
            if (candOut[p + d] < pk)
              atomicMax(&candOut[p + d], pk);
          }
        }
      }
    }
  }
  /* buffer is memset 0 per chunk: a zero verdict needs no write,
     and skipping it saves ~2MB of atomic traffic per chunk */
  if (out) atomicMax(&candOut[p], out);
  ZGPU_FLUSHK4();
#undef ZGPU_INS
#undef ZGPU_FLUSHK4
}

/* ---------------- kernel 2: sequential pick -----------------------
 * One warp per segment of ONE chunk (blockDim = segs*32, single
 * block).  Lane k prefetches the candidate for window position
 * base+k; lane 0 emits sequences via greedy + lazy rules over ONLY
 * the positions that have a candidate (bit-iteration over the
 * ballot mask - most window slots are empty).  Emitted records are
 * staged in shared memory and flushed coalesced - lane-0 global
 * stores are latency-bound.
 */
#define ZGPU_SHBUF 64
#define ZGPU_WARPS 8           /* warps per block - spread pick
                                  work over multiple SMs instead of
                                  pinning a whole chunk to one SM */
#define ZGPU_MAXSEGS 1024
__global__ void zgpu_pick_kernel(
    const u8 *s,
    u32 n,
    u32 segsPerChunk,
    const u32 *cand,
    const u32 *candK,
    u8 *seqOut,
    u32 *seqCounts,
    u32 *segCovs,
    u32 segSeqStride)
{
  __shared__ ZSeq shb[ZGPU_WARPS][ZGPU_SHBUF];
  __shared__ u32  wnd[ZGPU_WARPS][32];    /* per-warp candidates */
  __shared__ uint4 wndK[ZGPU_WARPS][32];  /* per-warp top-4 rows */
  u32 si = blockIdx.x * ZGPU_WARPS + (threadIdx.x >> 5);
  u32 wi = threadIdx.x >> 5;              /* local shared-mem row */
  u32 lane = threadIdx.x & 31;
  if (si >= segsPerChunk || si >= ZGPU_MAXSEGS) return;

  u32 segLen = (n + segsPerChunk - 1) / segsPerChunk;
  u32 lo = si * segLen;
  u32 hi = min(n, lo + segLen);
  if (lo >= hi) {
    if (lane == 0) { seqCounts[si] = 0; segCovs[si] = 0; }
    return;
  }

  ZSeq *seqs = (ZSeq *)(seqOut + (u64)si * segSeqStride);
  u32 seqCap = segSeqStride / sizeof(ZSeq);
  u32 anchor = lo, nSeq = 0, nLoc = 0;
  /* running rep-offset table (zstd initial state {1,4,8}); used only
     to PREFER repcode-friendly candidates - the encoder resolves the
     actual repcode, so imperfect per-segment state is safe */
  u32 rep0 = 1, rep1 = 4, rep2 = 8;
  int done = 0;

#define ZGPU_FLUSH() do { \
    u32 nl_ = __shfl_sync(ZGPU_FULLMASK, nLoc, 0); \
    u32 ns_ = __shfl_sync(ZGPU_FULLMASK, nSeq, 0); \
    __syncwarp(); \
    if (nl_) { \
      ZSeq *d_ = seqs + (ns_ - nl_); \
      for (u32 j_ = lane; j_ < nl_; j_ += 32) d_[j_] = shb[wi][j_]; \
      __syncwarp(); \
      if (lane == 0) nLoc = 0; \
    } \
  } while (0)

  for (u32 base = lo; base < hi && !done; ) {
    /* jump straight to the window containing the anchor: windows a
       long match covers carry no candidates for us anyway */
    u32 a = __shfl_sync(ZGPU_FULLMASK, anchor, 0);
    {
      u32 w = lo + ((a - lo) & ~31u);
      if (w > base)
        base = w;
    }
    if (base >= hi)
      break;
    u32 p = base + lane;
    u32 cnd = (p < hi) ? cand[p] : 0;
    unsigned valid = __ballot_sync(ZGPU_FULLMASK,
        (cnd >> 24) >= ZGPU_MINMATCH && p >= a && p < hi);
    /* parallel lazy evaluation: for every position holding a SHORT
       candidate, all lanes check their own forward-8 neighbourhood
       for a strictly better candidate - in parallel, one pass over
       shared memory instead of a serial scan per emit on lane 0.
       repcode-awareness stays on lane 0 (its bits are ignored when
       a candidate turns out rep-friendly). */
    unsigned deferMask = 0;
    if (valid) {
      wnd[wi][lane] = cnd;
      wndK[wi][lane] =
          (candK && p < hi) ? *(const uint4 *)(candK + (size_t)p * 4)
                            : make_uint4(0, 0, 0, 0);
      __syncwarp();
      {
      u32 len2 = cnd >> 24;
      int d = 0;
      if (len2 >= ZGPU_MINMATCH && len2 <= 16 && p >= a && p < hi) {
        uint4 r0 = wndK[wi][lane];
        u32 sc0 = 0;
#define ZGPU_SC0(v_) do { \
        u32 l_ = (v_) >> 24; \
        u32 o_ = (v_) & 0xFFFFFFu; \
        u32 s_ = (l_ << 3) - (o_ ? 31 - __clz(o_) : 24); \
        if (s_ > sc0) sc0 = s_; \
        } while (0)
        ZGPU_SC0(cnd); ZGPU_SC0(r0.x); ZGPU_SC0(r0.y);
        ZGPU_SC0(r0.z); ZGPU_SC0(r0.w);
#undef ZGPU_SC0
        u32 lim = lane + 9;
        for (u32 j = lane + 1; j < lim; j++) {
          /* score the best of the neighbour's own top-4 row plus its
             merged candidate - the lazy verdict should compare each
             position's strongest parse, not just the len-major one */
          u32 bs = 0;
          u32 jc = (j < 32) ? wnd[wi][j]
                            : ((base + j < hi) ? cand[base + j] : 0);
#define ZGPU_SCORE(v_) do { \
          u32 nl_ = (v_) >> 24; \
          u32 no_ = (v_) & 0xFFFFFFu; \
          u32 s_ = (nl_ << 3) - (no_ ? 31 - __clz(no_) : 24); \
          if (s_ > bs) bs = s_; \
          } while (0)
          ZGPU_SCORE(jc);
          if (j < 32) {
            uint4 rk_ = wndK[wi][j];
            ZGPU_SCORE(rk_.x); ZGPU_SCORE(rk_.y);
            ZGPU_SCORE(rk_.z); ZGPU_SCORE(rk_.w);
          }
#undef ZGPU_SCORE
          if ((int)(bs - sc0) > 24) { d = 1; break; }
        }
      }
      deferMask = __ballot_sync(ZGPU_FULLMASK, d);
      }
    }

    if (valid == 0) {
      /* nothing to emit in this window: positions just accumulate
         as literals on lane 0 (bail out of long incompressible
         runs so we stop wasting work on random data) */
      if (lane == 0 && !done) {
        u32 end = min(base + 32, hi);
        if (a < end && (int)(end - anchor) > 32768)
          done = 1;
      }
      done = __shfl_sync(ZGPU_FULLMASK, done, 0);
      base += 32;
      continue;
    }

    /* lane 0 walks the valid bits LOCALLY: literals, takes, emits,
       rep rolls and covered-bit clears all happen in registers/smem
       with no warp traffic - the old version paid two shfl round
       trips per candidate.  The walk escapes to the warp only for a
       capped-length take that needs cooperative re-extension, or a
       terminal condition; afterwards lane 0's `valid` is broadcast
       back so the while-loop guard stays uniform. */
    while (valid) {
      u32 sig = 0, extK = 0, extPk = 0;
      if (lane == 0 && !done) {
        u32 m = valid;
        while (m) {
          u32 k = (u32)(__ffs((int)m) - 1);
          u32 c2 = wnd[wi][k];
          u32 p2 = base + k;
          if (nSeq >= seqCap || (int)(p2 - anchor) > 32768) {
            sig = 2; break;
          }
          /* rep-aware best-of-5: the candK row holds up to 4
             alternative candidates the len-major atomicMax merge
             discarded; the merged entry itself is added back as a
             5th (offers only reach cand, not the row).  A repcode
             offset costs ~2 code bits vs the full high-bit price -
             rescore every packed (len|off) candidate and let the
             winner be a slightly shorter but free-offset match. */
          u32 pk = 0; int bsc = -2147483647;
          {
            uint4 rk = wndK[wi][k];
#define ZGPU_KPICK(v_) do { \
            u32 l_ = (v_) >> 24; \
            if (l_ >= ZGPU_MINMATCH) { \
              u32 o_ = (v_) & 0xFFFFFFu; \
              int sc_ = (int)((l_ << 3) - \
                  ((o_ == rep0 || o_ == rep1 || o_ == rep2) \
                      ? 2u : (o_ ? 31u - (u32)__clz(o_) : 24u))); \
              if (sc_ > bsc) { bsc = sc_; pk = (v_); } } \
          } while (0)
            ZGPU_KPICK(rk.x); ZGPU_KPICK(rk.y);
            ZGPU_KPICK(rk.z); ZGPU_KPICK(rk.w); ZGPU_KPICK(c2);
#undef ZGPU_KPICK
          }
          u32 len = pk >> 24;
          if (len < ZGPU_MINMATCH) { m &= m - 1; continue; }
          u32 off = pk & 0xFFFFFFu;
          /* backward extension: if the same-offset match also holds
             one byte earlier, slide the start back - the literal
             run shrinks and the match lengthens for free.  anchor
             only guards the window side; off <= p2 by construction */
          while (p2 > anchor && p2 - 1 >= off &&
                 s[p2 - 1] == s[p2 - 1 - off]) {
            p2--; len++;
          }
          /* the stored len was verified at the OLD start; after the
             backward slide the far end can run further - bounded
             scalar re-extend (cheap: mismatches exit at once) */
          {
            u32 flim = p2 + len + 16;
            if (flim > hi) flim = hi;
            while (p2 + len < flim &&
                   s[p2 + len] == s[p2 + len - off])
              len++;
          }
          int isRep = (off == rep0 || off == rep1 || off == rep2);
          /* value gate: a short match at a far offset costs more
             offset-code bits than it saves in literals */
          int skip = (!isRep && len <= 8 &&
              (len << 3) <= (31 - __clz(off)));
          /* lazy defer precomputed warp-wide above (rep-blind) */
          if (len <= 16 && !isRep && ((deferMask >> k) & 1))
            skip = 1;
          if (skip) { m &= m - 1; continue; }
          if (len >= ZGPU_LENCAP)
            { sig = 1; extK = k; extPk = pk; break; }
          /* roll the rep table: off becomes most-recent */
          if (off != rep0) {
            rep2 = rep1; rep1 = rep0; rep0 = off;
          }
          shb[wi][nLoc].off  = off;
          shb[wi][nLoc].lit  = p2 - anchor;
          shb[wi][nLoc].mlen = len;
          shb[wi][nLoc].rep  = 0;
          nLoc++; nSeq++;
          anchor = p2 + len;
          {
            u32 cl = len > 32 ? 32 : len;
            m &= ~((cl == 32 ? 0xFFFFFFFFu : ((1u << cl) - 1)) << k);
          }
        }
        valid = m;
      }
      {
        u32 w2 = __shfl_sync(ZGPU_FULLMASK,
            sig | (extK << 8), 0);
        sig = w2 & 0xFF;
        extK = w2 >> 8;
        valid = __shfl_sync(ZGPU_FULLMASK, valid, 0);
        extPk = __shfl_sync(ZGPU_FULLMASK, extPk, 0);
      }
      if (sig == 2) { done = 1; break; }
      if (sig == 0) break;
      /* cooperative extension for the capped take - extPk carries
         lane 0's rep-aware choice, which need not equal cand[extK] */
      {
        u32 off = extPk & 0xFFFFFFu;
        u32 p2 = base + extK;
        /* same backward slide as the lane-0 path: every lane runs
           the identical serial check (uniform, no divergence) so
           warp_extend sees the already-slid start */
        u32 a2 = __shfl_sync(ZGPU_FULLMASK, anchor, 0);
        while (p2 > a2 && p2 - 1 >= off &&
               s[p2 - 1] == s[p2 - 1 - off])
          p2--;
        u32 len = warp_extend(s, p2, off, ZGPU_LENCAP, hi, lane);
        if (lane == 0) {
          if (off != rep0) {
            rep2 = rep1; rep1 = rep0; rep0 = off;
          }
          shb[wi][nLoc].off  = off;
          shb[wi][nLoc].lit  = p2 - anchor;
          shb[wi][nLoc].mlen = len;
          shb[wi][nLoc].rep  = 0;
          nLoc++; nSeq++;
          anchor = p2 + len;
          {
            u32 cl = len > 32 ? 32 : len;
            valid &= ~((cl == 32 ? 0xFFFFFFFFu
                                   : ((1u << cl) - 1)) << extK);
          }
        }
        valid = __shfl_sync(ZGPU_FULLMASK, valid, 0);
      }
    }
    __syncwarp();
    /* end-of-window bookkeeping on lane 0 only; flush early enough
       that the next window's <=32 emits can never overflow shb */
    {
      int sig2 = 0;
      if (lane == 0) {
        if (nSeq >= seqCap) sig2 = 2;
        else if (nLoc > ZGPU_SHBUF - 33) sig2 = 1;
      }
      sig2 = __shfl_sync(ZGPU_FULLMASK, sig2, 0);
      if (sig2 == 2) done = 1;
      else if (sig2 == 1) ZGPU_FLUSH();
    }
    base += 32;
  }

  ZGPU_FLUSH();

  if (lane == 0) {
    seqCounts[si] = nSeq;
    segCovs[si]   = anchor - lo;
  }
#undef ZGPU_FLUSH
}

/* ------------- kernel 2b: optimal-parse pick (forward DP) -------
 * One THREAD per segment; the output contract is identical to
 * zgpu_pick_kernel (seqs/counts/covs) so every downstream stage is
 * untouched.  Three phases:
 *
 *   forward relaxation: state per position = {cost, rep0, rep1,
 *     rep2} - the rep table of the cheapest path to reach it (the
 *     zstd_opt model).  Edges: literal i->i+1 at litCost; match
 *     i->i+len at a rep-aware seqCost.  A 128-slot ring in local
 *     memory covers the <=63-deep lookahead; the slot consumed at
 *     position i is recycled for i+RING.
 *
 *   backward walk: replay the winning arrivals in choiceBuf from
 *     hi back to lo, collecting match triples.
 *
 *   forward emit: emit records in order; each match is re-extended
 *     to its true length (the DP priced it at <=63 B because the
 *     candidate packing caps len - extension only adds coverage to
 *     a seq whose cost is already paid).
 */
#define ZGPU_OPT_RING 128
__global__ void zgpu_opt_kernel(
    const u8 *__restrict__ s,
    u32 n,
    u32 segsPerChunk,
    const u32 *__restrict__ cand,
    const u32 *__restrict__ candK,
    u8 *__restrict__ seqOut,
    u32 *__restrict__ seqCounts,
    u32 *__restrict__ segCovs,
    u32 *__restrict__ choiceBuf,
    u32 segSeqStride,
    u32 litCost,
    u32 seqBase,
    u32 repC0,
    u32 repC1,
    u32 repC2)
{
  /* one THREAD per segment: the DP recurrence is inherently serial,
     so warp-cooperative staging would leave 31 lanes idle - instead
     every thread owns a segment and parallelism comes from the
     thousands of segments in flight across the pipeline ring */
  u32 si = blockIdx.x * blockDim.x + threadIdx.x;
  if (si >= segsPerChunk || si >= ZGPU_MAXSEGS) return;

  u32 segLen = (n + segsPerChunk - 1) / segsPerChunk;
  u32 lo = si * segLen;
  u32 hi = min(n, lo + segLen);
  if (lo >= hi) { seqCounts[si] = 0; segCovs[si] = 0; return; }

  /* forward-relaxation DP (zstd_opt style): each position's state
     carries {cost, rep0, rep1, rep2} - the rep table of the path
     that reached it cheapest.  Transitions are <= EXTCAP so a
     128-slot ring covers the whole lookahead: the slot freed by
     consuming position i becomes the target for i+RING.  Winning
     arrivals write their edge into choiceBuf for the replay. */
  struct OptS { u32 c, r0, r1, r2; };
  struct OptS st[ZGPU_OPT_RING];
  _Pragma("unroll")
  for (int t = 0; t < ZGPU_OPT_RING; t++) {
    st[t].c = 0xFFFFFFFFu;
    st[t].r0 = 1; st[t].r1 = 4; st[t].r2 = 8;
  }
  st[lo & (ZGPU_OPT_RING - 1)].c = 0;         /* segment start */

  for (u32 i = lo; i < hi; i++) {
    const u32 im = i & (ZGPU_OPT_RING - 1);
    struct OptS cur = st[im];
    st[im].c = 0xFFFFFFFFu;                  /* slot -> i + RING */
    /* literal edge i -> i+1: rep table passes through unchanged */
    {
      u32 nc = cur.c == 0xFFFFFFFFu ? cur.c : cur.c + litCost;
      struct OptS *d = &st[(i + 1) & (ZGPU_OPT_RING - 1)];
      if (nc < d->c) {
        d->c = nc; d->r0 = cur.r0; d->r1 = cur.r1; d->r2 = cur.r2;
        choiceBuf[i + 1] = 0;
      }
    }
    /* match edges */
    u32 c0 = cand[i];
    uint4 rk = *(const uint4 *)(candK + (size_t)i * 4);
    if (c0 | rk.x | rk.y | rk.z | rk.w) {
      u32 c5[5];
      c5[0] = c0; c5[1] = rk.x; c5[2] = rk.y;
      c5[3] = rk.z; c5[4] = rk.w;
      _Pragma("unroll")
      for (int j = 0; j < 5; j++) {
        u32 cd = c5[j];
        u32 len = cd >> 24;
        if (len < ZGPU_MINMATCH) continue;
        u32 off = cd & 0xFFFFFFu;
        if (!off || off > i) continue;
        if (len > ZGPU_EXTCAP) len = ZGPU_EXTCAP;  /* ring depth */
        if (i + len > hi) len = hi - i;
        if (len < ZGPU_MINMATCH) continue;
        u32 sc, r0 = cur.r0, r1 = cur.r1, r2 = cur.r2;
        if (off == r0)
          sc = seqBase + repC0;               /* rep0 ~2 bits */
        else if (off == r1)
          { sc = seqBase + repC1; r1 = r0; r0 = off; }
        else if (off == r2)
          { sc = seqBase + repC2; r2 = r1; r1 = r0; r0 = off; }
        else {
          sc = seqBase + ((u32)(31 - __clz(off)) << 8)
              + (len >= 16 ? 256u : 0u)
              + (len >= 48 ? 256u : 0u);
          r2 = r1; r1 = r0; r0 = off;
        }
        u32 nc = cur.c == 0xFFFFFFFFu ? cur.c : cur.c + sc;
        struct OptS *d = &st[(i + len) & (ZGPU_OPT_RING - 1)];
        if (nc < d->c) {
          d->c = nc; d->r0 = r0; d->r1 = r1; d->r2 = r2;
          choiceBuf[i + len] = (off << 10) | ((len - 1) << 1) | 1u;
        }
      }
    }
  }



  /* backward walk: collect matches into the candK row space, which
     is dead once the DP finishes (emit list = 3 u32/match, capacity
     segLen*4/3 >> any reachable seq count) */
  u32 *tmp = (u32 *)(candK + (size_t)lo * 4);
  u32 j = hi, K = 0;
  while (j > lo) {
    u32 e = choiceBuf[j];
    if (!e) { j--; continue; }
    u32 off = e >> 10;
    u32 len = ((e >> 1) & 511) + 1;
    tmp[K * 3]     = j - len;
    tmp[K * 3 + 1] = off;
    tmp[K * 3 + 2] = len;
    K++;
    j -= len;
  }



  /* forward emit: triples are in reverse order; extend each match
     to its TRUE length (bounded only by hi).  If the DP covered a
     long match with several <=EXTCAP edges, the extension runs
     straight through the intermediate planned matches - they are
     skipped so one long seq is emitted instead of a chain of
     capped ones (this merging is what keeps the seq count near
     the pick kernel's, the DP only chooses WHERE matches go). */
  {
    ZSeq *seqs = (ZSeq *)(seqOut + (u64)si * segSeqStride);
    u32 seqCap = segSeqStride / sizeof(ZSeq);
    u32 prevEnd = lo, nSeq = 0;
    for (u32 k = K; k-- > 0 && nSeq < seqCap; ) {
      u32 pos = tmp[k * 3];
      u32 off = tmp[k * 3 + 1];
      u32 len = tmp[k * 3 + 2];
      if (pos < prevEnd) {
        /* partially inside the prior extension: emit only the
           uncovered tail (a match's subrange is still a match) */
        if (pos + len <= prevEnd) continue;
        len -= prevEnd - pos; pos = prevEnd;
        if (len < ZGPU_MINMATCH) continue;
      }
      u32 lim = hi - pos - len;
      if (lim)
        len += matchlen(s + pos + len, s + pos + len - off, lim);
      seqs[nSeq].off  = off;
      seqs[nSeq].lit  = pos - prevEnd;
      seqs[nSeq].mlen = len;
      seqs[nSeq].rep  = 0;
      nSeq++;
      prevEnd = pos + len;
    }
    seqCounts[si] = nSeq;
    segCovs[si]   = prevEnd - lo;
  }
}

/* ------------- partition-neighborhood matcher -------------------
 * Original alternative to bucketed hashing: partition every position
 * by a 16-bit hash of its first 4 bytes into contiguous ranges of a
 * sorted[] array (single-pass radix partition, order inside a
 * partition is arbitrary).  Every real >=4-byte match pair shares
 * the same h4, so ALL match sources for a position live in its
 * partition - no 4-deep eviction, the neighborhood depth is the
 * whole partition.  Neighbor eval reuses the exact k1 contract:
 * atomicMax candOut[p] with (len<<24)|off plus forward offers.
 * Stages: hist -> scan -> scatter -> neighbor-eval. */

#define ZGPU_PLOG 16
#define ZGPU_PNBR 16

static __device__ __forceinline__ u32 h4of(const u8 *s, u32 p)
{
  return (rd32(s + p) * 2654435761u) >> (32 - ZGPU_PLOG);
}

__global__ void zgpu_phist_kernel(const u8 *__restrict__ s, u32 n,
    u32 *__restrict__ hist)
{
  u32 p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p + 4 <= n)
    atomicAdd(&hist[h4of(s, p)], 1);
}

/* exclusive scan of the 64K counters; one block, shared-mem scan */
__global__ void zgpu_pscan_kernel(const u32 *__restrict__ hist,
    u32 *__restrict__ base)
{
  __shared__ u32 sh[256];
  u32 carry = 0;
  for (u32 blk = 0; blk < (1u << ZGPU_PLOG) / 256; blk++) {
    u32 i = blk * 256 + threadIdx.x;
    u32 v = hist[i];
    __syncthreads();
    sh[threadIdx.x] = v;
    __syncthreads();
    for (u32 d = 1; d < 256; d <<= 1) {
      u32 t = threadIdx.x >= d ? sh[threadIdx.x - d] : 0;
      __syncthreads();
      sh[threadIdx.x] += t;
    }
    base[i] = carry + sh[threadIdx.x] - v;
    carry += sh[255];
  }
}

__global__ void zgpu_pscat_kernel(const u8 *__restrict__ s, u32 n,
    const u32 *__restrict__ base, u32 *__restrict__ fill,
    u32 *__restrict__ sorted)
{
  u32 p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p + 4 <= n) {
    u32 h = h4of(s, p);
    sorted[base[h] + atomicAdd(&fill[h], 1)] = p;
  }
}

__global__ void __launch_bounds__(256) zgpu_pmatch_kernel(
    const u8 *__restrict__ s, u32 n, u32 segsPerChunk,
    const u32 *__restrict__ base, const u32 *__restrict__ cnt,
    const u32 *__restrict__ sorted, u32 *__restrict__ candOut,
    u32 nvalid)
{
  u32 q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= nvalid) return;
  u32 p = sorted[q];
  u32 segLen = (n + segsPerChunk - 1) / segsPerChunk;
  u32 hi = min(n, (p / segLen + 1) * segLen);
  if (p + 8 > hi) return;
  u64 pv = rd64(s + p);
  u32 h = h4of(s, p);
  u32 lo = base[h], hi2 = lo + cnt[h];
  u32 myOff = 0, myLen = 0, myScore = 0;
#define ZGPU_PEV(qq_) do { \
      u32 cc_ = sorted[qq_]; \
      if (cc_ < p) { \
        u32 ll_ = candLen64(s, p, cc_, pv, hi); \
        if (ll_) { \
          u32 oo_ = p - cc_; \
          u32 sc_ = (ll_ << 3) - (31 - __clz(oo_)); \
          if ((int)(sc_ - myScore) > 0 || \
              (sc_ == myScore && oo_ < myOff)) \
            { myLen = ll_; myOff = oo_; myScore = sc_; } \
        } \
      } } while (0)
#pragma unroll
  for (u32 d = 1; d <= ZGPU_PNBR; d++) {
    if (q + d < hi2) ZGPU_PEV(q + d);
    if (q >= lo + d) ZGPU_PEV(q - d);
  }
#undef ZGPU_PEV
  if (myLen) {
    myLen = zgpu_truelen(s, p, myOff, myLen, hi);
    u32 out = ((myLen < ZGPU_LENCAP ? myLen : ZGPU_LENCAP) << 24)
        | myOff;
    if (candOut[p] < out) atomicMax(&candOut[p], out);
    for (u32 d = 1; d <= 4 && myLen > d + 3; d++) {
      if (p + d >= hi) break;
      u32 ol = (myLen >= ZGPU_LENCAP) ? ZGPU_LENCAP : myLen - d;
      u32 pk = (ol << 24) | myOff;
      if (candOut[p + d] < pk) atomicMax(&candOut[p + d], pk);
    }
  }
}

extern "C" int zgpu_part_launch(
    const void *src, unsigned n, unsigned segsPerChunk,
    void *phist, void *pfill, void *pbase, void *psorted,
    void *candOut, cudaStream_t stream)
{
  if (n < 8) return 0;
  u32 nvalid = n - 3;
  unsigned grid = (nvalid + 255) / 256;
  u32 *hi_ = (u32 *)phist, *fi_ = (u32 *)pfill, *ba_ = (u32 *)pbase;
  zgpu_phist_kernel<<<grid, 256, 0, stream>>>((const u8 *)src, n, hi_);
  zgpu_pscan_kernel<<<1, 256, 0, stream>>>(hi_, ba_);
  cudaMemsetAsync(fi_, 0, (1u << ZGPU_PLOG) * 4, stream);
  zgpu_pscat_kernel<<<grid, 256, 0, stream>>>(
      (const u8 *)src, n, ba_, fi_, (u32 *)psorted);
  zgpu_pmatch_kernel<<<grid, 256, 0, stream>>>(
      (const u8 *)src, n, segsPerChunk, ba_, hi_, (const u32 *)psorted,
      (u32 *)candOut, nvalid);
  return (int)cudaGetLastError();
}

/* ---------------- kernel 3: pack per-seg sequences ----------------
 * Copies each segment's produced sequence list from its strided
 * slot into one contiguous range.  Destination offsets are the
 * prefix sum over seqCounts, computed inline (segs is tiny), so the
 * host never has to round-trip the offsets.
 */
__global__ void zgpu_pack_kernel(const u8 *seqIn, u8 *seqOut,
    const u32 *seqCounts, unsigned nSegs, u32 segStride)
{
  /* one thread per 16-byte sequence record: blockIdx.y = segment */
  unsigned seg = blockIdx.y;
  unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= nSegs || j >= seqCounts[seg]) return;
  u64 dst = 0;
  for (unsigned t = 0; t < seg; t++)
    dst += seqCounts[t];
  const uint4 *srcp = (const uint4 *)(seqIn + (u64)seg * segStride);
  uint4 *dstp = (uint4 *)(seqOut + dst * 16);
  dstp[j] = srcp[j];
}

extern "C" int zgpu_cand_launch(
    const void *src, unsigned n, unsigned segsPerChunk,
    unsigned hashLog, void *hashTab, void *candOut, void *candK,
    cudaStream_t stream)
{
  unsigned block = 512;
  if (candK)
    zgpu_cand_kernel<true><<<(n + block - 1) / block, block, 0,
        stream>>>(
        (const u8 *)src, n, segsPerChunk, hashLog, (u32 *)hashTab,
        (u32 *)candOut, (u32 *)candK);
  else
    zgpu_cand_kernel<false><<<(n + block - 1) / block, block, 0,
        stream>>>(
        (const u8 *)src, n, segsPerChunk, hashLog, (u32 *)hashTab,
        (u32 *)candOut, NULL);
  return (int)cudaGetLastError();
}

extern "C" int zgpu_pick_launch(
    const void *src, unsigned n, unsigned segsPerChunk,
    const void *candIn, const void *candKIn, void *seqOut,
    unsigned *seqCounts, unsigned *segCovs,
    unsigned segSeqStride, cudaStream_t stream)
{
  zgpu_pick_kernel<<<(segsPerChunk + ZGPU_WARPS - 1) / ZGPU_WARPS,
      ZGPU_WARPS * 32, 0, stream>>>(
      (const u8 *)src, n, segsPerChunk, (const u32 *)candIn,
      (const u32 *)candKIn,
      (u8 *)seqOut, seqCounts, segCovs, segSeqStride);
  return (int)cudaGetLastError();
}

extern "C" int zgpu_opt_launch(
    const void *src, unsigned n, unsigned segsPerChunk,
    const void *candIn, const void *candKIn, void *seqOut,
    unsigned *seqCounts, unsigned *segCovs, void *choiceBuf,
    unsigned segSeqStride, unsigned litCost, unsigned seqBase,
    unsigned repC0, unsigned repC1, unsigned repC2,
    cudaStream_t stream)
{
  unsigned block = 32;    /* warp-granular: any segs count fills */
  zgpu_opt_kernel<<<(segsPerChunk + block - 1) / block,
      block, 0, stream>>>(
      (const u8 *)src, n, segsPerChunk, (const u32 *)candIn,
      (const u32 *)candKIn, (u8 *)seqOut, seqCounts, segCovs,
      (u32 *)choiceBuf, segSeqStride, litCost, seqBase,
      repC0, repC1, repC2);
  return (int)cudaGetLastError();
}

extern "C" int zgpu_pack_launch(
    const void *seqIn, void *seqOut,
    const unsigned *seqCounts,
    unsigned nSegs, unsigned segStride, cudaStream_t stream)
{
  unsigned recCap = segStride / 16;   /* records per segment slot */
  dim3 grid((recCap + 255) / 256, nSegs);
  zgpu_pack_kernel<<<grid, 256, 0, stream>>>(
      (const u8 *)seqIn, (u8 *)seqOut, seqCounts, nSegs, segStride);
  return (int)cudaGetLastError();
}

/* ============== GPU zstd frame writer (entropy path) ==============
 * Fully on-device: sequences (packed ZSeq) + raw input in, complete
 * .zst frame out.  v1 emits raw literals + predefined-mode FSE
 * sequence streams - the constant FSE tables are built once on the
 * host from the spec default distributions and live in __constant__.
 *
 * Layout work per chunk:
 *   prep  (1 thread):   seq -> block split at 128KB coverage,
 *                       per-seq literal-dst offsets, per-block meta
 *   litcopy (parallel): scatter literals into each block scratch
 *   blkenc (1/block):   literals header + nseq hdr + FSE bitstream
 *   out    (parallel):  frame header + 3B block headers + section
 *                       compaction into the final stream
 */

#define ZGPU_BLKMAX   (128u << 10)      /* zstd decoded block cap   */
#define ZGPU_MAXBLK   8                 /* chunk <= 4MB -> <=8...   */

struct ZgpuCTab {
  u32 dNb[64];      /* symbolTT.deltaNbBits                  */
  int  dFind[64];   /* symbolTT.deltaFindState               */
  u16  tab[512];    /* tableU16, size 1<<log                 */
  u32  log;
};

struct ZgpuBlk {
  u32 seq0, seqN;      /* [seq0, seqN) records                 */
  u32 litBytes;        /* literal bytes incl. tail literals    */
  u32 litSrcSkip;      /* =0; literals always scatter via seqs  */
  u32 litBase;         /* litOfs[seq0] - literal scratch base  */
  u32 decStart, decEnd;
  u32 len;             /* assembled content length (out)       */
  u32 litHdr;
};

/* ---- spec constant tables (zstd_internal.h / compress_internal.h) */
static const u8  ZG_LL_BITS[36] = {
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   1,1,1,1,2,2,3,3, 4,6,7,8,9,10,11,12, 13,14,15,16 };
static const u8  ZG_ML_BITS[53] = {
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 1,1,1,1,2,2,3,3, 4,4,5,7,8,9,10,11,
   12,13,14,15,16 };
static const u8  ZG_LL_CODE[64] = {
   0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
   16,16,17,17,18,18,19,19, 20,20,20,20,21,21,21,21,
   22,22,22,22,22,22,22,22, 23,23,23,23,23,23,23,23,
   24,24,24,24,24,24,24,24, 24,24,24,24,24,24,24,24 };
static const u8  ZG_ML_CODE[128] = {
   0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
   16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
   32,32,33,33,34,34,35,35, 36,36,36,36,37,37,37,37,
   38,38,38,38,38,38,38,38, 39,39,39,39,39,39,39,39,
   40,40,40,40,40,40,40,40, 40,40,40,40,40,40,40,40,
   41,41,41,41,41,41,41,41, 41,41,41,41,41,41,41,41,
   42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42,
   42,42,42,42,42,42,42,42, 42,42,42,42,42,42,42,42 };
static const short ZG_LL_NORM[36] = {
   4,3,2,2,2,2,2,2, 2,2,2,2,2,1,1,1,
   2,2,2,2,2,2,2,2, 2,3,2,1,1,1,1,1,
   -1,-1,-1,-1 };
static const short ZG_ML_NORM[53] = {
   1,4,3,2,2,2,2,2, 2,1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,1, 1,1,1,1,1,1,-1,-1,
   -1,-1,-1,-1,-1 };
static const short ZG_OF_NORM[29] = {
   1,1,1,1,1,1,2,2, 2,1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,1, -1,-1,-1,-1,-1 };

__constant__ ZgpuCTab zgc_ll, zgc_ml, zgc_of;
__constant__ u8  zgc_llbits[36], zgc_mlbits[53];
__constant__ u8  zgc_llcode[64], zgc_mlcode[128];
/* decode bases - extra bits are written as (value - base), not
   value & mask, since bases are not all 2^bits-aligned */
static const u32 ZG_LL_BASE[36] = {
   0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
   16,18,20,22,24,28,32,40, 48,64,0x80,0x100,0x200,0x400,0x800,0x1000,
   0x2000,0x4000,0x8000,0x10000 };
static const u32 ZG_ML_BASE[53] = {
   3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,
   19,20,21,22,23,24,25,26, 27,28,29,30,31,32,33,34,
   35,37,39,41,43,47,51,59, 67,83,99,0x83,0x103,0x203,0x403,0x803,
   0x1003,0x2003,0x4003,0x8003,0x10003 };
__constant__ u32 zgc_llbase[36], zgc_mlbase[53];

/* host-side port of FSE_buildCTable for a fixed (predefined) norm */
static void zgpu_build_ctab(ZgpuCTab *ct, const short *norm,
    int maxSym, int log)
{
  const u32 tableSize = 1u << log;
  const u32 tableMask = tableSize - 1;
  const u32 step = (tableSize >> 1) + (tableSize >> 3) + 3;
  u8  tableSym[512];
  u32 cumul[64];
  int  highThreshold = (int)tableSize - 1;
  int  total = 0;
  int  s;
  cumul[0] = 0;
  for (s = 0; s <= maxSym; s++) {
    if (norm[s] == -1) {
      cumul[s + 1] = cumul[s] + 1;
      tableSym[highThreshold--] = (u8)s;
    } else {
      cumul[s + 1] = cumul[s] + (u32)norm[s];
    }
  }
  {
    u32 pos = 0;
    for (s = 0; s <= maxSym; s++) {
      int i, n = norm[s];
      for (i = 0; i < n; i++) {
        tableSym[pos] = (u8)s;
        pos = (pos + step) & tableMask;
        while (pos > (u32)highThreshold)
          pos = (pos + step) & tableMask;
      }
    }
  }
  {
    u32 u;
    for (u = 0; u < tableSize; u++) {
      u8 sm = tableSym[u];
      ct->tab[cumul[sm]++] = (u16)(tableSize + u);
    }
  }
  total = 0;
  for (s = 0; s <= maxSym; s++) {
    if (norm[s] <= 0) {
      /* norm==0 slots are never hit with predefined norms, but keep
         the legal filler anyway */
      if (norm[s] == 0)
        ct->dNb[s] = ((u32)(log + 1) << 16) - tableSize;
      else {
        ct->dNb[s] = ((u32)log << 16) - tableSize;
        ct->dFind[s] = total - 1;
        total++;
        continue;
      }
      ct->dFind[s] = 0;
      continue;
    }
    if (norm[s] == 1) {
      ct->dNb[s] = ((u32)log << 16) - tableSize;
      ct->dFind[s] = total - 1;
      total++;
      continue;
    }
    {
      u32 hb = 0; u32 v = (u32)norm[s] - 1;
      while (v >>= 1) hb++;
      u32 maxBitsOut = (u32)log - hb;
      ct->dNb[s] = (maxBitsOut << 16) -
          ((u32)norm[s] << maxBitsOut);
      ct->dFind[s] = total - (u32)norm[s];
      total += (u32)norm[s];
    }
  }
  ct->log = (u32)log;
}

extern "C" int zgpu_entropy_init(void)
{
  static int done = 0;
  if (done) return 0;
  static ZgpuCTab ctll, ctml, ctof;
  memset(&ctll, 0, sizeof ctll);
  memset(&ctml, 0, sizeof ctml);
  memset(&ctof, 0, sizeof ctof);
  zgpu_build_ctab(&ctll, ZG_LL_NORM, 35, 6);
  zgpu_build_ctab(&ctml, ZG_ML_NORM, 52, 6);
  zgpu_build_ctab(&ctof, ZG_OF_NORM, 28, 5);
  cudaMemcpyToSymbol(zgc_ll, &ctll, sizeof ctll);
  cudaMemcpyToSymbol(zgc_ml, &ctml, sizeof ctml);
  cudaMemcpyToSymbol(zgc_of, &ctof, sizeof ctof);
  cudaMemcpyToSymbol(zgc_llbits, ZG_LL_BITS, sizeof ZG_LL_BITS);
  cudaMemcpyToSymbol(zgc_mlbits, ZG_ML_BITS, sizeof ZG_ML_BITS);
  cudaMemcpyToSymbol(zgc_llcode, ZG_LL_CODE, sizeof ZG_LL_CODE);
  cudaMemcpyToSymbol(zgc_mlcode, ZG_ML_CODE, sizeof ZG_ML_CODE);
  cudaMemcpyToSymbol(zgc_llbase, ZG_LL_BASE, sizeof ZG_LL_BASE);
  cudaMemcpyToSymbol(zgc_mlbase, ZG_ML_BASE, sizeof ZG_ML_BASE);
  done = 1;
  return (int)cudaGetLastError();
}

/* ---- kernels ---------------------------------------------------- */

/* prep: one thread walks the seq list, splits into <=128KB-coverage
   blocks, emits per-seq literal dst offsets + decoded end pos */
__global__ void zgpu_eprep_kernel(
    const ZSeq *__restrict__ seqs, u32 nseq, u32 n,
    u32 *__restrict__ litOfs,   /* per-seq literal dst base  */
    u32 *__restrict__ posOfs,   /* per-seq decoded end pos   */
    ZgpuBlk *__restrict__ blks, u32 *__restrict__ nblk)
{
  u32 pos = 0, lit = 0;
  u32 b = 0, seq0 = 0, blkStart = 0, blkLit0 = 0;
  for (u32 i = 0; i < nseq; i++) {
    litOfs[i] = lit;
    lit += seqs[i].lit;
    pos += seqs[i].lit + seqs[i].mlen;
    posOfs[i] = pos;
    /* close the block when this seq's coverage crosses 128KB */
    if (pos - blkStart > ZGPU_BLKMAX) {
      ZgpuBlk *B = &blks[b++];
      B->seq0 = seq0; B->seqN = i;
      B->litBase = blkLit0; B->litBytes = litOfs[i] - blkLit0;
      B->decStart = blkStart; B->decEnd = posOfs[i - 1];
      seq0 = i; blkStart = posOfs[i - 1]; blkLit0 = litOfs[i];
    }
  }
  /* final block: coverage ends at n - includes trailing literals */
  ZgpuBlk *B = &blks[b++];
  B->seq0 = seq0; B->seqN = nseq;
  B->litBase = blkLit0;
  B->litBytes = lit - blkLit0 + (n - pos);   /* + tail literals */
  B->decStart = blkStart; B->decEnd = n;
  *nblk = b;
  /* literals header size per block (raw literals): SF 1B/2B/3B */
  for (u32 j = 0; j < b; j++) {
    u32 lb = blks[j].litBytes;
    blks[j].litHdr = lb < 32 ? 1 : (lb < 4096 ? 2 : 3);
  }
}

/* litcopy: one thread per seq copies its literals into the block's
   scratch at [8B headroom][lithdr][lits]; plus tail literals */
__global__ void zgpu_elit_kernel(
    const u8 *__restrict__ in,
    const ZSeq *__restrict__ seqs, u32 nseq,
    const u32 *__restrict__ litOfs, const u32 *__restrict__ posOfs,
    const ZgpuBlk *__restrict__ blks, const u32 *__restrict__ nblkP,
    u8 *__restrict__ escr, u32 escrStride)
{
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nseq) {
    u32 nblk = *nblkP;
    /* find its block (seq index ordered) */
    u32 b = 0;
    while (b + 1 < nblk && i >= blks[b + 1].seq0) b++;
    const ZSeq *sq = &seqs[i];
    u32 src = posOfs[i] - sq->lit - sq->mlen;
    u8 *dst = escr + (size_t)b * escrStride + 8 +
        blks[b].litHdr + (litOfs[i] - blks[b].litBase);
    for (u32 j = 0; j < sq->lit; j++) dst[j] = in[src + j];
    /* last seq of the block also owns the trailing literals */
    if (i + 1 == blks[b].seqN) {
      u32 tail = blks[b].decEnd - posOfs[i];
      dst += sq->lit;
      for (u32 j = 0; j < tail; j++) dst[j] = in[posOfs[i] + j];
    }
  }
}

/* bit writer: u64 accumulator, forward bytes (decoder reads the
   stream backward - write order only fixes bit order) */
struct ZBitW {
  u8  *p;
  u64  acc;
  u32  nb;
};

static __device__ __forceinline__ void zw_add(ZBitW *w, u64 v, u32 nb)
{
  if (nb == 0) return;
  w->acc |= (v & ((1ull << nb) - 1)) << w->nb;
  w->nb += nb;                /* stays <64: adds <=22 bits per call */
  while (w->nb >= 8) {
    *w->p++ = (u8)w->acc;
    w->acc >>= 8;
    w->nb -= 8;
  }
}
static __device__ __forceinline__ void zw_close(ZBitW *w)
{
  zw_add(w, 1, 1);                        /* endMark */
  if (w->nb) { *w->p++ = (u8)w->acc; w->acc = 0; w->nb = 0; }
}

static __device__ __forceinline__ u32 fse_init(const ZgpuCTab *ct,
    u32 sym)
{
  u32 nb = (ct->dNb[sym] + (1u << 15)) >> 16;
  u32 st = (nb << 16) - ct->dNb[sym];
  return ct->tab[(st >> nb) + ct->dFind[sym]];
}
static __device__ __forceinline__ void fse_enc(ZBitW *w,
    const ZgpuCTab *ct, u32 *state, u32 sym)
{
  u32 nb = (*state + ct->dNb[sym]) >> 16;
  zw_add(w, *state, nb);
  *state = ct->tab[(*state >> nb) + ct->dFind[sym]];
}
static __device__ __forceinline__ u32 ll_code(u32 lit)
{
  return lit > 63 ? 31 - __clz(lit) + 19 : zgc_llcode[lit];
}
static __device__ __forceinline__ u32 ml_code(u32 mlb)
{
  return mlb > 127 ? 31 - __clz(mlb) + 36 : zgc_mlcode[mlb];
}

/* blkenc: one thread per block - header writes + the serial FSE
   chain for its sequence range.  Block scratch layout:
   [0..8 headroom][lithdr][lits][seqsec]                       */
__global__ void zgpu_eblk_kernel(
    const ZSeq *__restrict__ seqs,
    ZgpuBlk *__restrict__ blks, const u32 *__restrict__ nblkP,
    u8 *__restrict__ escr, u32 escrStride)
{
  u32 b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= *nblkP) return;
  ZgpuBlk *B = &blks[b];
  u8 *o = escr + (size_t)b * escrStride + 8;
  /* literals header: raw type(0) */
  u32 lb = B->litBytes;
  if (B->litHdr == 1)      *o = (u8)(lb << 3);
  else if (B->litHdr == 2) { u16 t = (u16)(0x4u | (lb << 4));
                           *(u16 *)o = t; }
  else                   { u32 t = 0xCu | (lb << 4);
                           o[0]=(u8)t; o[1]=(u8)(t>>8); o[2]=(u8)(t>>16); }
  u8 *p = o + B->litHdr + lb;
  /* sequences header */
  u32 nseq = B->seqN - B->seq0;
  if (nseq == 0) {
    *p++ = 0;
  } else {
    if (nseq < 128) *p++ = (u8)nseq;
    else if (nseq < 0x7F00) {
      p[0] = (u8)((nseq >> 8) + 0x80); p[1] = (u8)nseq; p += 2;
    } else {
      p[0] = 0xFF; p[1] = (u8)(nseq - 0x7F00);
      p[2] = (u8)((nseq - 0x7F00) >> 8); p += 3;
    }
    *p++ = 0;               /* all-predefined modes */
    /* FSE bitstream: states init on LAST seq, encode backward */
    const ZSeq *sq = seqs + B->seq0;
    u32 last = nseq - 1;
    u32 stML = fse_init(&zgc_ml, ml_code(sq[last].mlen - 3));
    u32 stOF = fse_init(&zgc_of, 31 - __clz(sq[last].off + 3));
    u32 stLL = fse_init(&zgc_ll, ll_code(sq[last].lit));
    ZBitW w; w.p = p; w.acc = 0; w.nb = 0;
    { u32 llc = ll_code(sq[last].lit), mlc = ml_code(sq[last].mlen - 3);
      zw_add(&w, sq[last].lit - zgc_llbase[llc], zgc_llbits[llc]);
      /* ML_base is match-length based: extra = mlen - base */
      zw_add(&w, sq[last].mlen - zgc_mlbase[mlc], zgc_mlbits[mlc]); }
    zw_add(&w, sq[last].off + 3,  31 - __clz(sq[last].off + 3));
    for (u32 n = last; n-- > 0; ) {
      u32 llc = ll_code(sq[n].lit);
      u32 mlb = sq[n].mlen - 3;
      u32 mlc = ml_code(mlb);
      u32 ofb = sq[n].off + 3;
      u32 ofc = 31 - __clz(ofb);
      fse_enc(&w, &zgc_of, &stOF, ofc);
      fse_enc(&w, &zgc_ml, &stML, mlc);
      fse_enc(&w, &zgc_ll, &stLL, llc);
      zw_add(&w, sq[n].lit - zgc_llbase[llc], zgc_llbits[llc]);
      zw_add(&w, sq[n].mlen - zgc_mlbase[mlc], zgc_mlbits[mlc]);
      zw_add(&w, ofb,       ofc);
    }
    zw_add(&w, stML, zgc_ml.log);
    zw_add(&w, stOF, zgc_of.log);
    zw_add(&w, stLL, zgc_ll.log);
    zw_close(&w);
    p = w.p;
  }
  B->len = (u32)(p - o);
}

/* out: assemble frame - header, per-block 3B header + content copy.
   One thread handles the header chain; a grid copies payload. */
__global__ void zgpu_eout_kernel(
    const u8 *__restrict__ escr, u32 escrStride,
    const ZgpuBlk *__restrict__ blks, const u32 *__restrict__ nblkP,
    u32 n,
    const u8 *__restrict__ in,
    u8 *__restrict__ out, u32 *__restrict__ outLen)
{
  __shared__ u32 s_off[ZGPU_MAXBLK + 1];
  __shared__ int  s_raw[ZGPU_MAXBLK];
  u32 nblk = *nblkP;
  if (nblk > ZGPU_MAXBLK) nblk = ZGPU_MAXBLK;
  if (threadIdx.x == 0) {
    /* frame header: magic + FHD(single-seg, 4B FCS) + FCS */
    out[0] = 0x28; out[1] = 0xB5; out[2] = 0x2F; out[3] = 0xFD;
    out[4] = 0xA0;
    out[5] = (u8)n; out[6] = (u8)(n >> 8);
    out[7] = (u8)(n >> 16); out[8] = (u8)(n >> 24);
    u32 off = 9;
    for (u32 b = 0; b < nblk; b++) {
      const ZgpuBlk *B = &blks[b];
      if (b >= ZGPU_MAXBLK) break;
      u32 decLen = B->decEnd - B->decStart;
      int raw = B->len + 3 >= decLen;   /* incompressible -> raw blk */
      s_raw[b] = raw;
      u32 sz = raw ? decLen : B->len;
      s_off[b] = off;
      u32 hdr = (u32)(b + 1 == nblk) | (raw ? 0u : 2u << 1) | (sz << 3);
      out[off] = (u8)hdr; out[off+1] = (u8)(hdr>>8);
      out[off+2] = (u8)(hdr>>16);
      off += 3 + sz;
    }
    s_off[nblk] = off;
    *outLen = off;
  }
  __syncthreads();
  /* parallel copy: thread covers bytes across all block payloads */
  u32 total = s_off[nblk];
  for (u32 b = 0; b < nblk; b++) {
    const ZgpuBlk *B = &blks[b];
    u32 dst0 = s_off[b] + 3;
    if (s_raw[b]) {
      u32 sz = B->decEnd - B->decStart;
      for (u32 j = threadIdx.x; j < sz; j += blockDim.x)
        out[dst0 + j] = in[B->decStart + j];
    } else {
      for (u32 j = threadIdx.x; j < B->len; j += blockDim.x)
        out[dst0 + j] =
            escr[(size_t)b * escrStride + 8 + j];
    }
  }
  (void)total;
}

extern "C" int zgpu_frame_launch(
    const void *src, unsigned n,
    const void *seqPack, unsigned nseq,
    void *litOfs, void *posOfs, void *blks, void *nblkP,
    void *escr, unsigned escrStride,
    void *out, void *outLenP,
    cudaStream_t stream)
{
  zgpu_eprep_kernel<<<1, 1, 0, stream>>>(
      (const ZSeq *)seqPack, nseq, n, (u32 *)litOfs, (u32 *)posOfs,
      (ZgpuBlk *)blks, (u32 *)nblkP);
  zgpu_elit_kernel<<<(nseq + 255) / 256, 256, 0, stream>>>(
      (const u8 *)src, (const ZSeq *)seqPack, nseq,
      (const u32 *)litOfs, (const u32 *)posOfs,
      (const ZgpuBlk *)blks, (const u32 *)nblkP,
      (u8 *)escr, escrStride);
  zgpu_eblk_kernel<<<1, 32, 0, stream>>>(
      (const ZSeq *)seqPack, (ZgpuBlk *)blks, (const u32 *)nblkP,
      (u8 *)escr, escrStride);
  /* out needs nblk to size the frame; launch enough threads for the
     largest block payload */
  zgpu_eout_kernel<<<1, 256, 0, stream>>>(
      (const u8 *)escr, escrStride, (const ZgpuBlk *)blks,
      (const u32 *)nblkP, n, (const u8 *)src, (u8 *)out,
      (u32 *)outLenP);
  return (int)cudaGetLastError();
}
