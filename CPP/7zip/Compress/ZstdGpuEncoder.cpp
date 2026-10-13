// ZstdGpuEncoder.cpp - zstd encoder offloaded to the GPU via nvCOMP

#include "StdAfx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../Windows/System.h"

#include "ZstdGpuEncoder.h"
#include "NvGpuDevice.h"
#include "GpuHostShared.h"

#ifndef Z7_EXTRACT_ONLY
namespace NCompress {
namespace NZSTDGPU {

/* nvCOMP has no level parameter; the level selects the chunk size.
   Larger chunks keep more cross-block redundancy (better ratio),
   smaller chunks give the GPU more parallelism. */
static size_t ChunkSizeForLevel(Int32 level)
{
  if (level < 0) level = -level;   /* fast levels */
  if (level <= 0)  return 1u << 20;
  if (level <= 3)  return 512u << 10;
  if (level <= 9)  return 1u << 20;
  if (level <= 16) return 4u << 20;
  return 16u << 20;                /* nvCOMP hard limit: 16 MB */
}

/* compressed bytes per wave kept on host+device */
static const size_t kWaveBytes = (size_t)192 << 20;

static const unsigned kMaxSeqWorkers = 32;
/* segments per chunk parsed in parallel on the GPU; the chunk hash
   table is shared, so the effective match window stays the chunk */
static const unsigned kSegsPerChunk = 32;

/* segment count per chunk: more segments = more parallel serial
   parsers in the pick kernel at the cost of more match boundaries.
   Z7GPU_SEGS overrides for experiments (32..1024). */
static unsigned Z7Segs()
{
  static volatile LONG segsCache = -1;
  LONG v = segsCache;
  if (v < 0) {
    char b[16];
    DWORD l = GetEnvironmentVariableA("Z7GPU_SEGS", b, 16);
    unsigned s = kSegsPerChunk;
    if (l > 0 && l < 16) {
      int t = atoi(b);
      if (t >= 32 && t <= 1024) s = (unsigned)t;
    }
    InterlockedCompareExchange(&segsCache, (LONG)s, -1);
    v = s;
  }
  return (unsigned)v;
}

/* ---- out-of-process helper: shared view + fwd decls ------------- */
struct Z7HostShm {
  HANDLE map;
  BYTE *base;
  Z7GpuHostCtrl *ctrl;
  BYTE *slots;               /* base + 4096 */
  HANDLE evReady, evQuit;
  HANDLE evIn[64], evOut[64];
  DWORD nSlots;
  HANDLE proc;
  bool isDaemon;             /* proc = shared daemon: never kill   */
};

static HRESULT Z7HostSpawn(Z7HostShm &h, DWORD inst, DWORD gen,
    DWORD nSlots, size_t chunk, size_t maxOut, size_t seqStride,
    size_t slotBytes, size_t mapBytes, int level, unsigned nWorkers,
    UInt64 startTok, const wchar_t *exe);
static void Z7HostCleanup(Z7HostShm &h);

struct Z7HostPending {
  HANDLE th;
  Z7HostShm *h;          /* malloc'd, filled by the worker          */
  HRESULT hr;
  DWORD inst;
  DWORD nSlots;
  size_t chunk, maxOut, seqStride, slotBytes, mapBytes;
  int level;
  unsigned nWorkers;
  wchar_t exe[MAX_PATH];
};

static DWORD WINAPI Z7PrespawnThread(void *p)
{
  Z7HostPending *pp = (Z7HostPending *)p;
  pp->h = (Z7HostShm *)malloc(sizeof(Z7HostShm));
  if (!pp->h) { pp->hr = E_OUTOFMEMORY; return 0; }
  pp->hr = Z7HostSpawn(*pp->h, pp->inst, 0, pp->nSlots, pp->chunk,
      pp->maxOut, pp->seqStride, pp->slotBytes, pp->mapBytes,
      pp->level, pp->nWorkers, 0, pp->exe);
  return 0;
}

/* process-wide pending spawn: 7z may create several CEncoder
   instances; only one speculative helper should ever be in flight
   (two at once would just serialize on the CUDA init mutex anyway) */
static volatile LONG g_hostInstSeq = 0;
static volatile LONG g_pendingSlot = 0;   /* 0 free, 1 in use      */
static Z7HostPending *g_pending = NULL;

CEncoder::CEncoder():
  _ctx(NULL),
  _srcBuf(NULL),
  _dstBuf(NULL),
  _srcBufSize(ZSTD_CStreamInSize()),
  _dstBufSize(ZSTD_CStreamOutSize()),
  _processedIn(0),
  _processedOut(0),
  _numThreads(NWindows::NSystem::GetNumberOfProcessors()),
  _Level(ZSTD_CLEVEL_DEFAULT),
  unpackSize(0),
  _seqCtx(NULL),
  _seqCtxs(NULL),
  _numSeqCtxs(0),
  _chunkSize(0),
  _waveN(0),
  _pinnedBuf(NULL)
{
  _props.clear();

  /* Speculative helper spawn: z7gpuhost.exe spends ~90ms in CUDA
     init; starting it while 7z scans/sets up hides most of that.
     CodeGpuHostSeq adopts it when the parameters match. */
  char engBuf[16];
  DWORD engLen = GetEnvironmentVariableA("Z7GPU_ENGINE", engBuf,
      (DWORD)sizeof(engBuf));
  if (engLen && strcmp(engBuf, "seq") && strcmp(engBuf, "host"))
    return;                       /* cpu/nvcomp/seqi: no helper    */

  if (InterlockedCompareExchange(&g_pendingSlot, 1, 0) != 0)
    return;                       /* a prespawn is already running */
  Z7HostPending *pp = (Z7HostPending *)malloc(sizeof(Z7HostPending));
  if (!pp) { g_pendingSlot = 0; return; }
  memset(pp, 0, sizeof(*pp));

  pp->inst = (DWORD)InterlockedIncrement(&g_hostInstSeq) - 1;
  pp->chunk = ChunkSizeForLevel(ZSTD_CLEVEL_DEFAULT);
  {
    char eb[32];
    DWORD el = GetEnvironmentVariableA("Z7GPU_CHUNKKB", eb, 32);
    if (el > 0 && el < 32) {
      size_t ck = (size_t)atol(eb) << 10;
      if (ck >= (64u << 10) && ck < ((size_t)1 << 22))
        pp->chunk = ck;
    }
  }
  pp->nSlots = 16;
  {
    char sb[16];
    DWORD sl = GetEnvironmentVariableA("Z7GPU_SLOTS", sb, 16);
    if (sl > 0 && sl < 16) {
      int v = atoi(sb);
      if (v >= 4 && v <= 64) pp->nSlots = (DWORD)v;
    }
  }
  pp->maxOut = ZSTD_compressBound(pp->chunk);
  pp->seqStride = (size_t)Z7Segs() *
      ((pp->chunk / Z7Segs() / 4 + 64) * sizeof(ZSTD_Sequence));
  pp->slotBytes = ((sizeof(Z7GpuSlotHdr) + pp->chunk + pp->maxOut)
      + 63) & ~(size_t)63;
  pp->mapBytes = 4096 + (size_t)pp->nSlots * pp->slotBytes;
  pp->level = ZSTD_CLEVEL_DEFAULT;
  /* the helper's encode workers run in a separate process while the
     codec thread blocks on drain, so all physical cores are free:
     scale past _numThreads (default = mmt) when the box has more */
  unsigned nw = _numThreads;
  {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (si.dwNumberOfProcessors > nw) nw = si.dwNumberOfProcessors;
    char wb[8];
    DWORD wl = GetEnvironmentVariableA("Z7GPU_WORKERS", wb, 8);
    if (wl && wl < 8) nw = (unsigned)atol(wb);
  }
  if (nw > kMaxSeqWorkers) nw = kMaxSeqWorkers;
  if (nw < 1) nw = 1;
  pp->nWorkers = nw;

  HMODULE self = NULL;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
          (LPCWSTR)&Z7PrespawnThread, &self) || !self) {
    free(pp); return;
  }
  DWORD n = GetModuleFileNameW(self, pp->exe, (DWORD)MAX_PATH);
  if (!n || n >= MAX_PATH) { free(pp); return; }
  wchar_t *slash = wcsrchr(pp->exe, L'\\');
  if (!slash) { free(pp); return; }
  slash[1] = 0;
  wcscat_s(pp->exe, L"z7gpuhost.exe");
  if (GetFileAttributesW(pp->exe) == INVALID_FILE_ATTRIBUTES) {
    free(pp); return;
  }
  pp->hr = E_PENDING;
  pp->th = CreateThread(NULL, 0, Z7PrespawnThread, pp, 0, NULL);
  if (!pp->th) { free(pp); g_pendingSlot = 0; return; }
  /* publish last so a racing Code() sees a fully formed record */
  MemoryBarrier();
  g_pending = pp;
}

CEncoder::~CEncoder()
{
  /* clean up CUDA while the process is still healthy - the static
     CNvGpuDevice dtor runs inside CRT teardown where driver calls
     race ntdll heap teardown (intermittent 0xC0000374 fast-fail).
     On the host path nothing was ever initialized - Shutdown() is
     a no-op and _pinnedBuf stays NULL. */
  CNvGpuDevice::Instance()->Shutdown();
  if (_ctx) {
    ZSTD_freeCCtx(_ctx);
    MyFree(_srcBuf);
    MyFree(_dstBuf);
  }
  if (_seqCtx)
    ZSTD_freeCCtx(_seqCtx);
  if (_seqCtxs) {
    for (unsigned i = 0; i < _numSeqCtxs; i++)
      if (_seqCtxs[i])
        ZSTD_freeCCtx(_seqCtxs[i]);
    MyFree(_seqCtxs);
  }
  if (_pinnedBuf)
    cudaHostUnregister(_pinnedBuf);   /* in-process path only */
}

size_t CEncoder::ChunkSize() const
{
  return _chunkSize ? _chunkSize : ChunkSizeForLevel(_Level);
}

Z7_COM7F_IMF(CEncoder::SetCoderProperties(const PROPID *propIDs,
    const PROPVARIANT *coderProps, UInt32 numProps))
{
  _props.clear();

  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPVARIANT &prop = coderProps[i];
    PROPID propID = propIDs[i];
    UInt32 v = (UInt32)prop.ulVal;
    switch (propID)
    {
    case NCoderPropID::kNumThreads:
      SetNumberOfThreads(v);
      break;
    case NCoderPropID::kLevel:
      _Level = (Int32)v;
      if (_Level < 1) _Level = 1;
      if (_Level > ZSTD_maxCLevel()) _Level = ZSTD_maxCLevel();
      _props._level = (Byte)_Level;
      break;
    default:
      break;
    }
  }
  return S_OK;
}

Z7_COM7F_IMF(CEncoder::SetCoderPropertiesOpt(const PROPID *propIDs,
    const PROPVARIANT *coderProps, UInt32 numProps))
{
  for (UInt32 i = 0; i < numProps; i++)
  {
    const PROPVARIANT &prop = coderProps[i];
    if (propIDs[i] == NCoderPropID::kExpectedDataSize)
      if (prop.vt == VT_UI8 && prop.uhVal.QuadPart)
        unpackSize = prop.uhVal.QuadPart;
  }
  return S_OK;
}

Z7_COM7F_IMF(CEncoder::WriteCoderProperties(ISequentialOutStream *outStream))
{
  return WriteStream(outStream, &_props, sizeof(_props));
}

/* -------- CPU fallback (no CUDA device / nvCOMP missing) -------- */
HRESULT CEncoder::CodeCpu(ISequentialInStream *inStream,
    ISequentialOutStream *outStream, ICompressProgressInfo *progress)
{
  ZSTD_EndDirective todo = ZSTD_e_continue;
  ZSTD_outBuffer outBuff;
  ZSTD_inBuffer inBuff;
  size_t err, srcSize;

  if (!_ctx) {
    _ctx = ZSTD_createCCtx();
    if (!_ctx)
      return E_OUTOFMEMORY;
    _srcBuf = MyAlloc(_srcBufSize);
    if (!_srcBuf)
      return E_OUTOFMEMORY;
    _dstBuf = MyAlloc(_dstBufSize);
    if (!_dstBuf)
      return E_OUTOFMEMORY;

    err = ZSTD_CCtx_setParameter(_ctx, ZSTD_c_compressionLevel, _Level);
    if (ZSTD_isError(err)) return E_INVALIDARG;
    err = ZSTD_CCtx_setParameter(_ctx, ZSTD_c_nbWorkers, (int)_numThreads);
    if (ZSTD_isError(err)) return E_INVALIDARG;
    err = ZSTD_CCtx_setParameter(_ctx, ZSTD_c_contentSizeFlag, 1);
    if (ZSTD_isError(err)) return E_INVALIDARG;
    if (unpackSize && unpackSize != (UInt64)(Int64)-1) {
      err = ZSTD_CCtx_setParameter(_ctx, ZSTD_c_srcSizeHint,
          (int)(unpackSize <= (UInt64)INT_MAX ? unpackSize : (UInt64)INT_MAX));
      if (ZSTD_isError(err)) return E_INVALIDARG;
    }
  }

  for (;;) {
    srcSize = _srcBufSize;
    RINOK(ReadStream(inStream, _srcBuf, &srcSize))
    if (srcSize == 0)
      todo = ZSTD_e_end;
    _processedIn += srcSize;

    for (;;) {
      outBuff.dst = _dstBuf;
      outBuff.size = _dstBufSize;
      outBuff.pos = 0;
      inBuff.src = _srcBuf;
      inBuff.size = srcSize;
      inBuff.pos = 0;

      err = ZSTD_compressStream2(_ctx, &outBuff, &inBuff, todo);
      if (ZSTD_isError(err))
        return E_FAIL;

      if (outBuff.pos) {
        RINOK(WriteStream(outStream, _dstBuf, outBuff.pos))
        _processedOut += outBuff.pos;
      }
      if (progress)
        RINOK(progress->SetRatioInfo(&_processedIn, &_processedOut))

      if (todo == ZSTD_e_end && err == 0)
        return S_OK;
      if (inBuff.pos == inBuff.size)
        break;
    }
  }
}

/* -------- GPU path -------- */
HRESULT CEncoder::CodeGpu(ISequentialInStream *inStream,
    ISequentialOutStream *outStream, ICompressProgressInfo *progress)
{
  CNvGpuDevice *gpu = CNvGpuDevice::Instance();

  const size_t chunk = ChunkSize();
  unsigned waveN = (unsigned)(kWaveBytes / chunk);
  if (waveN < 8) waveN = 8;
  if (waveN > 1024) waveN = 1024;

  size_t maxOut = 0;
  if (nvcompBatchedZstdCompressGetMaxOutputChunkSize(chunk,
          nvcompBatchedZstdCompressDefaultOpts, &maxOut) != nvcompSuccess)
    return E_FAIL;

  try {
    _inBuf.Alloc(waveN * chunk);
    _outBuf.Alloc(waveN * maxOut);
    _sizes.Alloc(waveN);
  } catch (...) {
    return E_OUTOFMEMORY;
  }
  if (!_inBuf.NonConstData() || !_outBuf.NonConstData() || !_sizes.NonConstData())
    return E_OUTOFMEMORY;

  for (;;) {
    /* fill staging buffer; chunk-aligned except the tail */
    size_t filled = 0;
    while (filled < waveN * chunk) {
      size_t want = waveN * chunk - filled;
      RINOK(ReadStream(inStream, _inBuf + filled, &want))
      if (want == 0)
        break;
      filled += want;
    }
    if (filled == 0) {
      if (_processedIn == 0) {
        /* empty input: emit a canonical empty zstd frame via CPU so
           stock decoders (which reject a 0-byte stream) stay happy */
        return CodeCpu(inStream, outStream, progress);
      }
      return S_OK; /* eof */
    }

    unsigned n = (unsigned)((filled + chunk - 1) / chunk);
    size_t off = 0;
    for (unsigned i = 0; i < n; i++) {
      size_t cs = filled - off;
      if (cs > chunk) cs = chunk;
      _sizes[i] = cs;
      off += cs;
    }

    gpu->Lock();
    HRESULT hr = gpu->CompressWave(_inBuf, _sizes.NonConstData(), n,
        maxOut, _outBuf, _sizes.NonConstData());
    gpu->Unlock();
    if (FAILED(hr))
      return hr;

    for (unsigned i = 0; i < n; i++) {
      RINOK(WriteStream(outStream, _outBuf + i * maxOut, _sizes[i]))
      _processedOut += _sizes[i];
    }
    _processedIn += filled;
    if (progress)
      RINOK(progress->SetRatioInfo(&_processedIn, &_processedOut))

    if (filled < waveN * chunk)
      return S_OK; /* short read => eof */
  }
}

/* -------- GPU path: custom CUDA sequence generator + zstd framing ----

   The CUDA kernel performs the LZ77 parse; ZSTD_compressSequences()
   turns the sequences into a standard zstd frame on the CPU.  If the
   sequence path ever rejects a chunk, that chunk is recompressed with
   plain ZSTD_compress() (still a valid standalone frame).

   Chunk-granular pipeline: the device holds a ring of K independent
   stream slots (see NvGpuDevice.cpp).  The main thread reads chunk i,
   submits its GPU chain, drains the slot being reused (stitching its
   per-segment sequences into one list) and releases one semaphore
   token per stitched chunk.  A pool of workers pulls token+index
   pairs and runs ZSTD_compressSequences() as soon as a chunk's
   sequences land - so GPU seqgen, host stitching and entropy coding
   all run concurrently.
------------------------------------------------------------------- */

/* fold per-segment GPU output into a single ZSTD_Sequence list:
   each segment's trailing literals roll into the next segment's
   first litLength field */
static unsigned StitchSegs(const Byte *pack, const unsigned *counts,
    const unsigned *covs, size_t n, unsigned segs, Byte *dst)
{
  unsigned wpos = 0;
  unsigned pending = 0;
  unsigned packOff = 0;
  const size_t segLen = (n + segs - 1) / segs;
  for (unsigned s = 0; s < segs; s++) {
    const unsigned cnt = counts[s];
    const size_t lo = (size_t)s * segLen;
    const size_t hi = lo + segLen < n ? lo + segLen : n;
    const Byte *raw = pack + (size_t)packOff * 16;
    packOff += cnt;
    for (unsigned j = 0; j < cnt; j++) {
      unsigned rec[4];
      memcpy(rec, raw + (size_t)j * 16, 16);
      if (j == 0) {
        rec[1] += pending;
        pending = 0;
      }
      memcpy(dst + (size_t)wpos * 16, rec, 16);
      wpos++;
    }
    pending += (unsigned)((hi - lo) - covs[s]);
  }
  return wpos;
}

struct CSeqEncShared
{
  volatile LONG next;
  volatile unsigned n;      /* UINT_MAX until wave drain completes */
  HANDLE ready;
  volatile LONG finUs;      /* SeqPipeFinish wait total (us) */
  volatile LONG stitchUs;   /* StitchSegs total            */
  volatile LONG encUs;      /* ZSTD_compressSequences total */
};

/* bulk reader: fills chunk i of _inBuf ahead of the pipeline loop so
   the producer never serializes disk latency against GPU waits */
struct CSeqReadJob
{
  ISequentialInStream *inStream;
  Byte *dst;
  size_t *sizes;
  size_t chunk;
  unsigned maxN;          /* buffer bound: never read past maxN     */
  unsigned n;             /* written on exit: chunks actually read */
  HRESULT hr;
  HANDLE ready;           /* +1 per chunk available */
  volatile LONG *stop;
};

static THREAD_FUNC_DECL SeqReadWorker(void *p)
{
  CSeqReadJob *j = (CSeqReadJob *)p;
  j->hr = S_OK;
  j->n = 0;
  for (unsigned i = 0; i < j->maxN; i++) {
    if (*j->stop)
      break;
    size_t got = 0;
    while (got < j->chunk) {
      size_t want = j->chunk - got;
      HRESULT hr = ReadStream(j->inStream,
          j->dst + (size_t)i * j->chunk + got, &want);
      if (FAILED(hr)) { j->hr = hr; break; }
      if (want == 0) break;
      got += want;
    }
    if (FAILED(j->hr) || got == 0)
      break;
    j->sizes[i] = got;
    j->n = i + 1;
    ReleaseSemaphore(j->ready, 1, NULL);
    if (got < j->chunk)          /* short read => eof */
      break;
  }
  /* final token so the consumer can observe eof/error instead of
     waiting forever */
  ReleaseSemaphore(j->ready, 1, NULL);
  return 0;
}

struct CSeqEncJob
{
  CNvGpuDevice *gpu;
  unsigned K;
  unsigned segs;
  ZSTD_CCtx *ctx;
  const Byte *src;
  const size_t *sizes;
  Byte *seqs;
  unsigned *counts;           /* per-chunk stitched seq counts */
  Byte *dst;
  size_t *outSizes;
  size_t chunk;
  size_t seqStride;
  size_t maxOut;
  int level;
  CSeqEncShared *sh;
};

static THREAD_FUNC_DECL SeqEncWorker(void *p)
{
  CSeqEncJob *j = (CSeqEncJob *)p;
  CSeqEncShared *sh = j->sh;
  for (;;) {
    WaitForSingleObject(sh->ready, INFINITE);
    unsigned i = (unsigned)InterlockedIncrement(&sh->next) - 1;
    if (i >= sh->n) {
      ReleaseSemaphore(sh->ready, 1, NULL);   /* hand the token on */
      return 0;
    }
    /* slot i%K deterministically holds chunk i: the producer hands
       the index to the slot ring in order.  Finish() blocks on the
       slot's pack D2H, then we stitch, free the slot for reuse and
       run the entropy coding. */
    const unsigned s = i % j->K;
    HANDLE hFree = j->gpu->SeqPipeFreeEvent(s);
    const int dbg =
        GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0;
    LONGLONG t0 = 0, t1 = 0, t2 = 0;
    if (dbg) {
      LARGE_INTEGER q;
      QueryPerformanceCounter(&q);
      t0 = q.QuadPart;
    }
    HRESULT hr = j->gpu->SeqPipeFinish(s);
    if (dbg) {
      LARGE_INTEGER q;
      QueryPerformanceCounter(&q);
      t1 = q.QuadPart;
    }
    if (FAILED(hr)) {
      j->outSizes[i] = (size_t)-1;
      SetEvent(hFree);
      continue;
    }
    j->counts[i] = StitchSegs(j->gpu->SeqPipePacked(s),
        j->gpu->SeqPipeCounts(s), j->gpu->SeqPipeCovs(s),
        j->sizes[i], j->segs,
        j->seqs + (size_t)i * j->seqStride);
    SetEvent(hFree);   /* slot's host data extracted - reusable */
    if (dbg) {
      LARGE_INTEGER q;
      QueryPerformanceCounter(&q);
      t2 = q.QuadPart;
    }

    const size_t cs = j->sizes[i];
    const Byte *src = j->src + (size_t)i * j->chunk;
    const ZSTD_Sequence *sq = (const ZSTD_Sequence *)(const void *)
        (j->seqs + (size_t)i * j->seqStride);
    size_t r = ZSTD_compressSequences(j->ctx,
        j->dst + (size_t)i * j->maxOut, j->maxOut,
        sq, j->counts[i], src, cs);
    if (ZSTD_isError(r))
      r = ZSTD_compress(j->dst + (size_t)i * j->maxOut, j->maxOut,
          src, cs, j->level);
    j->outSizes[i] = ZSTD_isError(r) ? (size_t)-1 : r;
    if (dbg) {
      LARGE_INTEGER q, f;
      QueryPerformanceCounter(&q);
      QueryPerformanceFrequency(&f);
      LONGLONG u = f.QuadPart / 1000;   /* ticks per ms */
      InterlockedAdd((LONG *)&sh->finUs,
          (LONG)(((t1 - t0) * 1000) / u));
      InterlockedAdd((LONG *)&sh->stitchUs,
          (LONG)(((t2 - t1) * 1000) / u));
      InterlockedAdd((LONG *)&sh->encUs,
          (LONG)(((q.QuadPart - t2) * 1000) / u));
    }
  }
}

HRESULT CEncoder::CodeGpuSeq(ISequentialInStream *inStream,
    ISequentialOutStream *outStream, ICompressProgressInfo *progress)
{
  CNvGpuDevice *gpu = CNvGpuDevice::Instance();

  size_t chunk = ChunkSize();
  bool chunkForced = false;
  {  /* testing hook: Z7GPU_CHUNKKB overrides the seq chunk size */
    char eb[32];
    DWORD el = GetEnvironmentVariableA("Z7GPU_CHUNKKB", eb, 32);
    if (el > 0 && el < 32) {
      size_t ck = (size_t)atol(eb) << 10;
      if (ck >= (64u << 10) && ck < ((size_t)1 << 22)) {
        chunk = ck;
        chunkForced = true;
      }
    }
  }
  unsigned waveN = (unsigned)(kWaveBytes / chunk);
  if (waveN < 8) waveN = 8;
  if (waveN > 128) waveN = 128;   /* seq buffers are ~4x input */

  /* mutable: the adaptive-window retune below recomputes all of
     them at a wave boundary */
  size_t maxOut = ZSTD_compressBound(chunk);
  /* per-segment capacity: worst case ~1 seq per MINMATCH bytes, the
     region is Z7Segs() wide and stitched into one list/chunk */
  size_t seqStride = (size_t)Z7Segs() *
      ((chunk / Z7Segs() / 4 + 64) * sizeof(ZSTD_Sequence));

  unsigned wlog = ZSTD_WINDOWLOG_MIN;
  while (((size_t)1 << wlog) < chunk && wlog < ZSTD_WINDOWLOG_MAX)
    wlog++;

  unsigned nWorkers = _numThreads;
  if (nWorkers > kMaxSeqWorkers) nWorkers = kMaxSeqWorkers;
  if (nWorkers < 1) nWorkers = 1;

  if (_numSeqCtxs < nWorkers) {
    if (_seqCtxs) {
      for (unsigned i = 0; i < _numSeqCtxs; i++)
        if (_seqCtxs[i]) ZSTD_freeCCtx(_seqCtxs[i]);
      MyFree(_seqCtxs);
      _seqCtxs = NULL;
      _numSeqCtxs = 0;
    }
    _seqCtxs = (ZSTD_CCtx **)MyAlloc(nWorkers * sizeof(ZSTD_CCtx *));
    if (!_seqCtxs)
      return E_OUTOFMEMORY;
    memset(_seqCtxs, 0, nWorkers * sizeof(ZSTD_CCtx *));
    for (unsigned i = 0; i < nWorkers; i++) {
      ZSTD_CCtx *c = ZSTD_createCCtx();
      if (!c)
        return E_OUTOFMEMORY;
      size_t err;
      err = ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, _Level);
      if (ZSTD_isError(err)) return E_INVALIDARG;
      err = ZSTD_CCtx_setParameter(c, ZSTD_c_minMatch, ZSTD_MINMATCH_MIN);
      if (ZSTD_isError(err)) return E_INVALIDARG;
      err = ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, (int)wlog);
      if (ZSTD_isError(err)) return E_INVALIDARG;
      err = ZSTD_CCtx_setParameter(c, ZSTD_c_contentSizeFlag, 1);
      if (ZSTD_isError(err)) return E_INVALIDARG;
      {  /* match the helper path: opt-in safety net, off by default */
        char vb[8];
        DWORD vl = GetEnvironmentVariableA("Z7GPU_VALIDATE", vb, 8);
        err = ZSTD_CCtx_setParameter(c, ZSTD_c_validateSequences,
            (vl && vb[0] == '1') ? 1 : 0);
      }
      if (ZSTD_isError(err)) return E_INVALIDARG;
      /* our sequences carry no rep info; let zstd resolve repcodes
         (off by default below level 10, worth it for our matcher) */
      err = ZSTD_CCtx_setParameter(c, ZSTD_c_repcodeResolution, 1);
      if (ZSTD_isError(err)) return E_INVALIDARG;
      _seqCtxs[i] = c;
    }
    _numSeqCtxs = nWorkers;
  }

  try {
    _inBuf.Alloc(waveN * chunk);
    _outBuf.Alloc(waveN * maxOut);
    _sizes.Alloc(waveN);
    _outSizes.Alloc(waveN);
    _seqBuf.Alloc(waveN * seqStride);
    _seqCounts.Alloc(waveN);
  } catch (...) {
    return E_OUTOFMEMORY;
  }
  if (!_inBuf.NonConstData() || !_outBuf.NonConstData() ||
      !_sizes.NonConstData() || !_outSizes.NonConstData() ||
      !_seqBuf.NonConstData() || !_seqCounts.NonConstData())
    return E_OUTOFMEMORY;

  const bool dbg = GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0;
  /* adaptive window: when Z7GPU_CHUNKKB is unset the first wave runs
     at 512KB; if it compresses below ~6% the data is highly
     compressible (text-like) and the remaining waves are retuned to
     a 2MB chunk for a 4x deeper match window - worth ~-20% size on
     text, while repetitive binary corpora (>=10%) keep the faster
     512KB layout. */
  bool adaptiveDone = chunkForced || chunk > ((size_t)512 << 10);

  gpu->Lock();
  HRESULT hrInit = gpu->SeqPipeInit(chunk, Z7Segs(), seqStride);
  gpu->Unlock();
  if (FAILED(hrInit))
    return hrInit;
  unsigned K = gpu->SeqPipeDepth();
  if (K == 0 || K > 64)
    return E_FAIL;

  /* pin the H2D staging buffer: without this the driver stages
     pageable memory through an internal pinned buffer per copy */
  if ((void *)_inBuf != _pinnedBuf) {
    if (_pinnedBuf &&
        cudaHostUnregister(_pinnedBuf) != cudaSuccess)
      (void)cudaGetLastError();
    _pinnedBuf = NULL;
    {
      cudaError_t re = cudaHostRegister(_inBuf.NonConstData(),
          waveN * chunk, cudaHostRegisterDefault);
      if (re == cudaSuccess)
        _pinnedBuf = (void *)_inBuf;
      else {
        if (dbg)
          fprintf(stderr, "[zstdgpu]   hostRegister=%d %s\n", (int)re,
              cudaGetErrorString(re));
        (void)cudaGetLastError();   /* clear the sticky error code */
      }
    }
  }

  CSeqEncJob jobs[kMaxSeqWorkers];
  CThread threads[kMaxSeqWorkers];
  unsigned threadsMade = 0;
  bool eof = false;

  for (;;) {
    /* ---- one wave: produce chunks, drain ring, encode concurrently */
    CSeqEncShared sh;
    sh.next = 0;
    sh.n = 0xFFFFFFFFu;
    sh.finUs = sh.stitchUs = sh.encUs = 0;
    sh.ready = CreateSemaphoreW(NULL, 0, 0x7FFFFFFF, NULL);
    if (!sh.ready)
      return E_FAIL;

    unsigned w = nWorkers;
    threadsMade = 0;
    for (unsigned i = 0; i < w; i++) {
      CSeqEncJob *j = &jobs[i];
      j->gpu = gpu;
      j->K = K;
      j->segs = Z7Segs();
      j->ctx = _seqCtxs[i];
      j->src = _inBuf;
      j->sizes = _sizes.NonConstData();
      j->seqs = _seqBuf;
      j->counts = _seqCounts.NonConstData();
      j->dst = _outBuf;
      j->outSizes = _outSizes.NonConstData();
      j->chunk = chunk;
      j->seqStride = seqStride;
      j->maxOut = maxOut;
      j->level = _Level;
      j->sh = &sh;
      if (Thread_Create(&threads[threadsMade], SeqEncWorker, j) == 0)
        threadsMade++;
    }

    /* bulk reader fills _inBuf ahead; +1 token per chunk plus one
       final token when it stops */
    HANDLE readSem = CreateSemaphoreW(NULL, 0, 0x7FFFFFFF, NULL);
    if (!readSem) {
      CloseHandle(sh.ready);
      return E_FAIL;
    }
    volatile LONG readStop = 0;
    CSeqReadJob rj;
    rj.inStream = inStream;
    rj.dst = _inBuf;
    rj.sizes = _sizes.NonConstData();
    rj.chunk = chunk;
    rj.maxN = waveN;
    rj.n = 0;
    rj.hr = S_OK;
    rj.ready = readSem;
    rj.stop = &readStop;
    CThread readThread;
    bool readerUp = Thread_Create(&readThread, SeqReadWorker, &rj) == 0;

    DWORD tk0 = GetTickCount();
    unsigned n = 0;
    unsigned tWaitFree = 0, tRead = 0;
    HRESULT hr = S_OK;

    gpu->Lock();
    gpu->SeqPipeReset();
    gpu->Unlock();
    for (; n < waveN && SUCCEEDED(hr); ) {
      const unsigned s = n % K;
      /* wait for the reader to land chunk n */
      if (readerUp) {
        DWORD tq2 = dbg ? GetTickCount() : 0;
        WaitForSingleObject(readSem, INFINITE);
        if (dbg) tRead += GetTickCount() - tq2;
        if (FAILED(rj.hr)) {
          if (dbg)
            fprintf(stderr, "[zstdgpu]   read-hr=0x%08X at n=%u\n",
                (unsigned)rj.hr, n);
          hr = rj.hr; break;
        }
        if (n >= rj.n) { eof = true; break; }
      } else {
        /* reader thread failed to spawn: read inline */
        size_t got = 0;
        while (got < chunk) {
          size_t want = chunk - got;
          hr = ReadStream(inStream,
              _inBuf + (size_t)n * chunk + got, &want);
          if (FAILED(hr) || want == 0) break;
          got += want;
        }
        if (FAILED(hr)) break;
        if (got == 0) { eof = true; break; }
        _sizes[n] = got;
        if (got < chunk) eof = true;
      }
      /* wait until the slot's previous consumer finished stitching */
      DWORD tq0 = dbg ? GetTickCount() : 0;
      WaitForSingleObject(gpu->SeqPipeFreeEvent(s), INFINITE);
      if (dbg) tWaitFree += GetTickCount() - tq0;
      hr = gpu->SeqPipeSubmit(s, _inBuf + (size_t)n * chunk,
          _sizes[n]);
      if (FAILED(hr)) break;
      if (_sizes[n] < chunk)
        eof = true;
      n++;
      ReleaseSemaphore(sh.ready, 1, NULL);
    }

    /* stop + join the reader (it may be ahead of us) */
    InterlockedExchange(&readStop, 1);
    if (readerUp)
      Thread_Wait_Close(&readThread);
    CloseHandle(readSem);

    /* stop the workers: n submitted chunks each got a token; poison
       the semaphore so every waiter observes the end and leaves */
    sh.n = n;
    ReleaseSemaphore(sh.ready, w, NULL);
    for (unsigned i = 0; i < threadsMade; i++)
      Thread_Wait_Close(&threads[i]);
    CloseHandle(sh.ready);
    /* degenerate case: no worker could spawn - encode inline */
    if (threadsMade == 0 && SUCCEEDED(hr)) {
      CSeqEncJob *j = &jobs[0];
      for (unsigned i = 0; i < n; i++) {
        const size_t cs = j->sizes[i];
        const Byte *src = j->src + (size_t)i * j->chunk;
        const ZSTD_Sequence *sq = (const ZSTD_Sequence *)(const void *)
            (j->seqs + (size_t)i * j->seqStride);
        size_t r = ZSTD_compressSequences(j->ctx,
            j->dst + (size_t)i * j->maxOut, j->maxOut,
            sq, j->counts[i], src, cs);
        if (ZSTD_isError(r))
          r = ZSTD_compress(j->dst + (size_t)i * j->maxOut,
              j->maxOut, src, cs, j->level);
        j->outSizes[i] = ZSTD_isError(r) ? (size_t)-1 : r;
      }
    }
    if (dbg)
      fprintf(stderr,
          "[zstdgpu] wave n=%u pipe=%ums (waitfree=%u read=%u "
          "fin=%ld stitch=%ld enc=%ld)\n",
          n, GetTickCount() - tk0, tWaitFree, tRead,
          sh.finUs / 1000, sh.stitchUs / 1000, sh.encUs / 1000);

    if (FAILED(hr)) {
      fprintf(stderr, "[zstdgpu] ABORT hr=0x%08lX n=%u eof=%d\n",
          (unsigned long)hr, n, (int)eof);
      gpu->Lock();
      gpu->SeqPipeReset();
      gpu->Unlock();
      return hr;
    }

    DWORD tk1 = GetTickCount();
    for (unsigned i = 0; i < n; i++) {
      if (_outSizes[i] == (size_t)-1)
        return E_FAIL;
      RINOK(WriteStream(outStream, _outBuf + i * maxOut, _outSizes[i]))
      _processedOut += _outSizes[i];
    }
    size_t inWave = 0;
    for (unsigned i = 0; i < n; i++)
      inWave += _sizes[i];
    _processedIn += inWave;
    if (progress)
      RINOK(progress->SetRatioInfo(&_processedIn, &_processedOut))
    if (dbg)
      fprintf(stderr, "[zstdgpu]   write=%ums\n",
          GetTickCount() - tk1);

    if (n == 0 && _processedIn == 0)
      return CodeCpu(inStream, outStream, progress); /* empty frame */
    if (eof || n < waveN)
      return S_OK;

    /* adaptive window: highly compressible first wave (<~6%)?  then
       retune the remaining waves to a deeper 2MB chunk.  All pending
       GPU/CPU work for this wave is done by now, so buffers and
       contexts can be rebuilt safely. */
    if (!adaptiveDone) {
      adaptiveDone = true;
      size_t inSum = 0, outSum = 0;
      for (unsigned i = 0; i < n; i++) {
        inSum += _sizes[i];
        outSum += _outSizes[i];
      }
      if (inSum >= ((size_t)4 << 20) && outSum * 16 < inSum) {
        chunk = (size_t)2 << 20;
        waveN = (unsigned)(kWaveBytes / chunk);
        if (waveN < 8) waveN = 8;
        if (waveN > 128) waveN = 128;
        maxOut = ZSTD_compressBound(chunk);
        seqStride = (size_t)Z7Segs() *
            ((chunk / Z7Segs() / 4 + 64) * sizeof(ZSTD_Sequence));
        unsigned wlog2 = ZSTD_WINDOWLOG_MIN;
        while (((size_t)1 << wlog2) < chunk &&
            wlog2 < ZSTD_WINDOWLOG_MAX)
          wlog2++;
        for (unsigned i = 0; i < nWorkers; i++) {
          ZSTD_freeCCtx(_seqCtxs[i]);
          ZSTD_CCtx *c = ZSTD_createCCtx();
          if (!c)
            return E_OUTOFMEMORY;
          ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, _Level);
          ZSTD_CCtx_setParameter(c, ZSTD_c_minMatch,
              ZSTD_MINMATCH_MIN);
          ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, (int)wlog2);
          ZSTD_CCtx_setParameter(c, ZSTD_c_contentSizeFlag, 1);
          {
            char vb[8];
            DWORD vl = GetEnvironmentVariableA("Z7GPU_VALIDATE",
                vb, 8);
            ZSTD_CCtx_setParameter(c, ZSTD_c_validateSequences,
                (vl && vb[0] == '1') ? 1 : 0);
          }
          ZSTD_CCtx_setParameter(c, ZSTD_c_repcodeResolution, 1);
          _seqCtxs[i] = c;
        }
        if (_pinnedBuf &&
            cudaHostUnregister(_pinnedBuf) != cudaSuccess)
          (void)cudaGetLastError();
        _pinnedBuf = NULL;
        try {
          _inBuf.Alloc(waveN * chunk);
          _outBuf.Alloc(waveN * maxOut);
          _sizes.Alloc(waveN);
          _outSizes.Alloc(waveN);
          _seqBuf.Alloc(waveN * seqStride);
          _seqCounts.Alloc(waveN);
        } catch (...) {
          return E_OUTOFMEMORY;
        }
        if (!_inBuf.NonConstData() || !_outBuf.NonConstData() ||
            !_sizes.NonConstData() || !_outSizes.NonConstData() ||
            !_seqBuf.NonConstData() || !_seqCounts.NonConstData())
          return E_OUTOFMEMORY;
        gpu->Lock();
        hrInit = gpu->SeqPipeInit(chunk, Z7Segs(), seqStride);
        K = gpu->SeqPipeDepth();
        gpu->Unlock();
        if (FAILED(hrInit))
          return hrInit;
        if (dbg)
          fprintf(stderr,
              "[zstdgpu] adaptive: wave %.1f%% -> %zuKB chunks\n",
              100.0 * (double)outSum / (double)inSum,
              chunk >> 10);
      }
    }
  }
}

/* diagnostic: report the faulting address/code instead of dying
   silently - removed once the intermittent init crash is solved */

/* walk PEB loader list without heap access to map an address to a module */
struct ZgpuUStr {
  unsigned short Length;
  unsigned short MaximumLength;
  const unsigned short *Buffer;
};
struct ZgpuLdrEntry {
  LIST_ENTRY LoadOrder;
  LIST_ENTRY MemOrder;
  LIST_ENTRY InitOrder;
  void *DllBase;
  void *EntryPoint;
  ULONG SizeOfImage;
  ZgpuUStr FullName;
  ZgpuUStr BaseName;
};
struct ZgpuPebLdr {
  ULONG Length;
  void *Reserved;
  LIST_ENTRY LoadOrder;
  LIST_ENTRY MemOrder;
  LIST_ENTRY InitOrder;
};
struct ZgpuPeb {
  UCHAR pad[0x18];
  ZgpuPebLdr *Ldr;
};

static const ZgpuLdrEntry *ZgpuFindModule(const void *addr,
    ZgpuUStr const **nameOut)
{
  const ZgpuPeb *peb = (const ZgpuPeb *)__readgsqword(0x60);
  if (!peb || !peb->Ldr)
    return NULL;
  LIST_ENTRY *head = &peb->Ldr->LoadOrder;
  for (LIST_ENTRY *e = head->Flink; e && e != head; e = e->Flink) {
    const ZgpuLdrEntry *ent = (const ZgpuLdrEntry *)e;
    const char *lo = (const char *)ent->DllBase;
    if ((const char *)addr >= lo &&
        (const char *)addr < lo + ent->SizeOfImage) {
      if (nameOut)
        *nameOut = &ent->BaseName;
      return ent;
    }
  }
  return NULL;
}

static void ZgpuPrintSym(const void *addr)
{
  ZgpuUStr const *name = NULL;
  const ZgpuLdrEntry *ent = ZgpuFindModule(addr, &name);
  if (!ent) {
    fprintf(stderr, "    %p -> ?\n", addr);
    return;
  }
  char nm[260];
  unsigned i = 0;
  const unsigned short *w = name->Buffer;
  unsigned nch = name->Length / 2;
  while (w && *w && i < sizeof(nm) - 1 && i < nch)
    nm[i++] = (char)(*w++ & 0xFF);
  nm[i] = 0;
  fprintf(stderr, "    %p -> %s+0x%llx\n", addr, nm,
      (unsigned long long)((const char *)addr - (const char *)ent->DllBase));
}

static LONG WINAPI ZgpuVeh(PEXCEPTION_POINTERS ep)
{
  fprintf(stderr,
      "[zstdgpu] CRASH code=0x%08lX addr=%p tid=%lu\n",
      (unsigned long)ep->ExceptionRecord->ExceptionCode,
      ep->ExceptionRecord->ExceptionAddress,
      (unsigned long)GetCurrentThreadId());
  ZgpuPrintSym(ep->ExceptionRecord->ExceptionAddress);
  {
    /* 7z headers shadow _EXCEPTION_RECORD; use raw layout */
    struct ZgpuExcRec {
      DWORD Code, Flags;
      void *Next;
      void *Address;
      DWORD NumParams, pad;
      ULONG_PTR Info[15];
    };
    const ZgpuExcRec *er = (const ZgpuExcRec *)ep->ExceptionRecord;
    fprintf(stderr, "    nparams=%lu flags=%lx\n",
        (unsigned long)er->NumParams, (unsigned long)er->Flags);
    for (ULONG i = 0; i < er->NumParams && i < 15; i++)
      fprintf(stderr, "    param[%lu]=%p\n", (unsigned long)i,
          (void *)er->Info[i]);
    /* deep-dump only for the heap-corruption signature: param[0] of
       any OTHER exception (e.g. 0xE06D7363's EH magic) is not a valid
       pointer, and dereferencing it crashes inside this handler */
    /* C++ exception: identify the thrown type via the object's
       vtable pointer (module+offset of the ??_7 symbol) */
    if (er->Code == 0xE06D7363 && er->NumParams >= 2 && er->Info[1]) {
      MEMORY_BASIC_INFORMATION mbx;
      const void *obj = (const void *)er->Info[1];
      if (VirtualQuery(obj, &mbx, sizeof(mbx)) &&
          !(mbx.Protect & (PAGE_NOACCESS | PAGE_GUARD))) {
        fprintf(stderr, "    excobj vtbl -> ");
        ZgpuPrintSym(*(void *const *)obj);
      }
    }
    if (er->Code == 0xC0000374 && er->NumParams >= 1 && er->Info[0]) {
      MEMORY_BASIC_INFORMATION mb;
      const unsigned long long *q = (const unsigned long long *)
          er->Info[0];
      if (VirtualQuery(q, &mb, sizeof(mb)) &&
          !(mb.Protect & (PAGE_NOACCESS | PAGE_GUARD))) {
        for (int i = 0; i < 24; i++)
          fprintf(stderr, "    info+%02x = %016llx\n",
              i * 8, q[i]);
        /* the freed block (info[3]) may still hold identifying
           content: dump 96 bytes as hex+ascii */
        const unsigned char *blk = (const unsigned char *)q[3];
        if (blk && VirtualQuery(blk, &mb, sizeof(mb)) &&
            !(mb.Protect & (PAGE_NOACCESS | PAGE_GUARD))) {
          for (int r = 0; r < 6; r++) {
            fprintf(stderr, "    blk+%02x:", r * 16);
            for (int c = 0; c < 16; c++)
              fprintf(stderr, " %02x", blk[r * 16 + c]);
            fprintf(stderr, "  ");
            for (int c = 0; c < 16; c++) {
              unsigned char ch = blk[r * 16 + c];
              fprintf(stderr, "%c", (ch >= 32 && ch < 127) ? ch : '.');
            }
            fprintf(stderr, "\n");
          }
        }
      }
    }
    CONTEXT *c = ep->ContextRecord;
    if (c)
      fprintf(stderr,
          "    rcx=%p rdx=%p r8=%p r9=%p rbx=%p rsi=%p rdi=%p\n",
          (void *)c->Rcx, (void *)c->Rdx, (void *)c->R8,
          (void *)c->R9, (void *)c->Rbx, (void *)c->Rsi,
          (void *)c->Rdi);
  }
  void *frames[48];
  USHORT n = RtlCaptureStackBackTrace(1, 48, frames, NULL);
  for (USHORT i = 0; i < n; i++)
    ZgpuPrintSym(frames[i]);
  /* identify WHICH heap is corrupt */
  {
    HANDLE heaps[64];
    DWORD nh = GetProcessHeaps(64, heaps);
    if (nh > 64) nh = 64;
    for (DWORD i = 0; i < nh; i++) {
      BOOL ok = HeapValidate(heaps[i], 0, NULL);
      fprintf(stderr, "    heap[%lu]=%p valid=%d\n",
          (unsigned long)i, heaps[i], (int)ok);
      if (!ok)
        fflush(stderr);
    }
    fprintf(stderr, "    procHeap=%p\n", GetProcessHeap());
  }
  fflush(stderr);
  return EXCEPTION_CONTINUE_SEARCH;
}

/* ------------------------------------------------------------------
   Out-of-process CUDA path.  cuInit/nvcuda64 intermittently fast-fails
   (0xC0000374 HEAP_FAILURE_INVALID_ENTRY) inside 7z.exe on driver
   616.92, so all CUDA work lives in z7gpuhost.exe.  We create the
   shared-memory ring + events, spawn the helper next to this module,
   stream raw chunks in and read standard zstd frames out.

   Reliability: helper startup failure falls back to the CPU path.
   A mid-stream helper death respawns a fresh helper (new generation
   of shm objects); pending chunk inputs are copied from the dead
   generation's slots and fed again - chunks are independent frames
   so the archive stays byte-correct.  Repeated respawns or a
   helper-side encode error return E_FAIL rather than garbage.
------------------------------------------------------------------ */

#define Z7_E_HOSTDIED E_ABORT   /* respawnable: helper process lost  */

static void Z7DbgLog(const char *fmt, ...)
{
  FILE *f = NULL;
  if (fopen_s(&f, "E:/dev/7z-cuda/z7dbg.log", "a") != 0 || !f)
    return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(f, fmt, ap);
  va_end(ap);
  fclose(f);
}

static void Z7HostClose(Z7HostShm &h)
{
  if (h.proc) { CloseHandle(h.proc); h.proc = NULL; }
  if (h.evReady) { CloseHandle(h.evReady); h.evReady = NULL; }
  if (h.evQuit) { CloseHandle(h.evQuit); h.evQuit = NULL; }
  for (DWORD i = 0; i < h.nSlots; i++) {
    if (h.evIn[i]) { CloseHandle(h.evIn[i]); h.evIn[i] = NULL; }
    if (h.evOut[i]) { CloseHandle(h.evOut[i]); h.evOut[i] = NULL; }
  }
  if (h.base) { UnmapViewOfFile(h.base); h.base = NULL; }
  if (h.map) { CloseHandle(h.map); h.map = NULL; }
  h.ctrl = NULL; h.slots = NULL;
}

static void Z7HostCleanup(Z7HostShm &h)
{
  if (h.evQuit) SetEvent(h.evQuit);
  /* a daemon outlives us by design: signal quit so it ends our
     session, but never wait on or kill the shared process */
  if (!h.isDaemon && h.proc &&
      WaitForSingleObject(h.proc, 4000) == WAIT_TIMEOUT)
    TerminateProcess(h.proc, 1);
  Z7HostClose(h);
}

/* daemon handoff: if a persistent z7gpuhost --daemon is accepting
   sessions, hand it this control-block name through the rendezvous
   channel (skipping ~110ms of CUDA init); spawn one if none lives.
   Returns true when the daemon accepted the session - h.proc then
   holds a SYNCHRONIZE handle on the daemon for death detection. */
static bool Z7HostDaemonHandoff(Z7HostShm &h, const char *name,
    const wchar_t *exe)
{
  if (GetEnvironmentVariableA("Z7GPU_NODAEMON", NULL, 0))
    return false;
  HANDLE evAlive = OpenEventA(SYNCHRONIZE, FALSE,
      Z7GPU_DAEMON_ALIVE);
  if (!evAlive) {
    /* no daemon - spawn one and give it a moment to create the
       rendezvous objects */
    wchar_t dcmd[MAX_PATH + 64];
    swprintf(dcmd, sizeof(dcmd) / sizeof(dcmd[0]),
        L"\"%s\" " L"" Z7GPU_DAEMON_ARG, exe);
    STARTUPINFOW si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));
    if (!CreateProcessW(exe, dcmd, NULL, NULL, FALSE,
            CREATE_NO_WINDOW, NULL, NULL, &si, &pi))
      return false;
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    DWORD t0 = GetTickCount();
    while (!evAlive && GetTickCount() - t0 < 3000) {
      Sleep(10);
      evAlive = OpenEventA(SYNCHRONIZE, FALSE, Z7GPU_DAEMON_ALIVE);
    }
    if (!evAlive) return false;
    DWORD wr = WaitForSingleObject(evAlive, 2000);
    CloseHandle(evAlive);
    if (wr != WAIT_OBJECT_0) return false;   /* busy bootstrapping */
  } else {
    DWORD wr = WaitForSingleObject(evAlive, 0);
    CloseHandle(evAlive);
    if (wr != WAIT_OBJECT_0) return false;   /* daemon busy */
  }
  /* serialize channel writes */
  HANDLE mtx = OpenMutexA(SYNCHRONIZE, FALSE, Z7GPU_DAEMON_MTX);
  if (!mtx || WaitForSingleObject(mtx, 1000) != WAIT_OBJECT_0) {
    if (mtx) CloseHandle(mtx);
    return false;
  }
  bool ok = false;
  HANDLE chMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE,
      Z7GPU_DAEMON_CHAN);
  BYTE *ch = chMap ? (BYTE *)MapViewOfFile(chMap,
      FILE_MAP_ALL_ACCESS, 0, 0, 4096) : NULL;
  HANDLE evReq = OpenEventA(EVENT_MODIFY_STATE, FALSE,
      Z7GPU_DAEMON_REQ);
  HANDLE evAck = OpenEventA(SYNCHRONIZE, FALSE, Z7GPU_DAEMON_ACK);
  HANDLE evAl2 = OpenEventA(SYNCHRONIZE, FALSE,
      Z7GPU_DAEMON_ALIVE);
  if (ch && evReq && evAck && evAl2 &&
      WaitForSingleObject(evAl2, 0) == WAIT_OBJECT_0) {
    ((volatile DWORD *)ch)[0] = GetCurrentProcessId();
    memset(ch + 8, 0, 160);
    memcpy(ch + 8, name, strlen(name));
    MemoryBarrier();
    SetEvent(evReq);
    if (WaitForSingleObject(evAck, 2000) == WAIT_OBJECT_0) {
      DWORD dpid = ((volatile DWORD *)ch)[1];
      if (dpid)
        h.proc = OpenProcess(SYNCHRONIZE, FALSE, dpid);
      ok = h.proc != NULL;
    }
  }
  if (ch) UnmapViewOfFile(ch);
  if (chMap) CloseHandle(chMap);
  if (evReq) CloseHandle(evReq);
  if (evAck) CloseHandle(evAck);
  if (evAl2) CloseHandle(evAl2);
  ReleaseMutex(mtx);
  CloseHandle(mtx);
  return ok;
}

/* create objects for generation `gen` of encoder instance `inst`,
   spawn helper, wait READY */
static HRESULT Z7HostSpawn(Z7HostShm &h, DWORD inst, DWORD gen,
    DWORD nSlots, size_t chunk, size_t maxOut, size_t seqStride,
    size_t slotBytes, size_t mapBytes, int level, unsigned nWorkers,
    UInt64 startTok, const wchar_t *exe)
{
  memset(&h, 0, sizeof(h));
  h.nSlots = nSlots;
  char name[160];
  wsprintfA(name, "%s%lu.%lu.%lu", Z7GPU_HOST_PREFIX,
      (unsigned long)GetCurrentProcessId(), (unsigned long)inst,
      (unsigned long)gen);
  char *base = name + strlen(name);

  h.map = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL,
      PAGE_READWRITE, (DWORD)(mapBytes >> 32),
      (DWORD)(mapBytes & 0xFFFFFFFFu), name);
  if (!h.map) return E_FAIL;
  h.base = (BYTE *)MapViewOfFile(h.map, FILE_MAP_ALL_ACCESS, 0, 0, 0);
  if (!h.base) { Z7HostClose(h); return E_FAIL; }
  memset(h.base, 0, mapBytes);
  h.ctrl = (Z7GpuHostCtrl *)(void *)h.base;
  h.slots = h.base + 4096;

  strcpy_s(base, sizeof(name) - (size_t)(base - name),
      Z7GPU_EVT_READY);
  h.evReady = CreateEventA(NULL, FALSE, FALSE, name);
  strcpy_s(base, sizeof(name) - (size_t)(base - name),
      Z7GPU_EVT_QUIT);
  h.evQuit = CreateEventA(NULL, TRUE, FALSE, name);
  for (DWORD s = 0; s < nSlots; s++) {
    wsprintfA(base, "%s%u", Z7GPU_EVT_IN, s);
    h.evIn[s] = CreateEventA(NULL, FALSE, FALSE, name);
    wsprintfA(base, "%s%u", Z7GPU_EVT_OUT, s);
    h.evOut[s] = CreateEventA(NULL, FALSE, FALSE, name);
  }
  if (!h.evReady || !h.evQuit) { Z7HostClose(h); return E_FAIL; }
  for (DWORD s = 0; s < nSlots; s++)
    if (!h.evIn[s] || !h.evOut[s]) { Z7HostClose(h); return E_FAIL; }

  /* publish config; helper gates on magic */
  h.ctrl->version = Z7GPU_HOST_VERSION;
  h.ctrl->nSlots = nSlots;
  h.ctrl->chunk = chunk;
  h.ctrl->segs = Z7Segs();
  h.ctrl->seqStride = seqStride;
  h.ctrl->maxOut = maxOut;
  h.ctrl->level = level;
  h.ctrl->nWorkers = nWorkers;
  h.ctrl->slotBytes = slotBytes;
  h.ctrl->startTok = startTok;
  MemoryBarrier();
  h.ctrl->magic = Z7GPU_HOST_MAGIC;
  MemoryBarrier();

  *base = 0;    /* strip the last event suffix left by the loop */

  /* daemon handoff first: a live daemon reuses its warm CUDA
     context; fall through to direct spawn on any failure */
  h.isDaemon = Z7HostDaemonHandoff(h, name, exe);
  if (!h.isDaemon) {
    wchar_t cmd[MAX_PATH + 260];
    wchar_t wname[160];
    MultiByteToWideChar(CP_ACP, 0, name, -1, wname, 160);
    swprintf(cmd, sizeof(cmd) / sizeof(cmd[0]),
        L"\"%s\" %lu %s", exe,
        (unsigned long)GetCurrentProcessId(), wname);
    STARTUPINFOW si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));
    if (!CreateProcessW(exe, cmd, NULL, NULL, FALSE,
            CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
      if (GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0))
        fprintf(stderr, "[zstdgpu] spawn err=%lu\n",
            (unsigned long)GetLastError());
      Z7HostClose(h);
      return E_FAIL;
    }
    CloseHandle(pi.hThread);
    h.proc = pi.hProcess;
  }

  DWORD t0 = GetTickCount();
  for (;;) {
    DWORD st = h.ctrl->state;
    if (st == Z7HOST_S_READY) {
      if (GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0) {
        fprintf(stderr, "[zstdgpu] host ready in %lums\n",
            (unsigned long)(GetTickCount() - t0));
        fflush(stderr);
      }
      return S_OK;
    }
    if (st == Z7HOST_S_FAILED) {
      if (GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0))
        fprintf(stderr, "[zstdgpu] host reported FAILED\n");
      break;
    }
    HANDLE w[2] = { h.evReady, h.proc };
    DWORD wr = WaitForMultipleObjects(2, w, FALSE, 500);
    if (wr == WAIT_OBJECT_0 + 1) {
      if (GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0)) {
        DWORD ec = 0; GetExitCodeProcess(h.proc, &ec);
        fprintf(stderr, "[zstdgpu] host died ec=0x%lx st=%lu\n",
            (unsigned long)ec, (unsigned long)st);
      }
      break;                                    /* helper died */
    }
    if (GetTickCount() - t0 > Z7GPU_READY_TIMEOUT_MS) break;
  }
  /* dead or failed helper - reap it hard so objects free up (never
     kill a daemon: it just failed our session) */
  if (!h.isDaemon && h.proc &&
      WaitForSingleObject(h.proc, 2000) == WAIT_TIMEOUT)
    TerminateProcess(h.proc, 1);
  Z7HostClose(h);
  return E_FAIL;
}

/* wait for chunk output on shm slot s; writes it to outStream.
   E_ABORT = helper died/timed out (respawnable), E_FAIL = encode error */
static HRESULT Z7HostDrain(Z7HostShm &h, unsigned s,
    ISequentialOutStream *outStream, CEncoder *enc)
{
  Z7GpuSlotHdr *sl = (Z7GpuSlotHdr *)(void *)
      (h.slots + (size_t)s * h.ctrl->slotBytes);
  HANDLE w[2] = { h.evOut[s], h.proc };
  DWORD wr = WaitForMultipleObjects(2, w, FALSE, 120000);
  if (wr != WAIT_OBJECT_0)
    return Z7_E_HOSTDIED;               /* timeout or helper died  */
  if (sl->err)
    return E_FAIL;
  const BYTE *out = (const BYTE *)(const void *)(sl + 1)
      + h.ctrl->chunk;
  RINOK(WriteStream(outStream, out, (size_t)sl->outLen))
  enc->_processedOut += sl->outLen;
  return S_OK;
}

HRESULT CEncoder::CodeGpuHostSeq(ISequentialInStream *inStream,
    ISequentialOutStream *outStream, ICompressProgressInfo *progress)
{
  const bool dbg = GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0;
  Z7HostShm h;
  memset(&h, 0, sizeof(h));
  HRESULT hr = S_OK;
  DWORD i = 0;                 /* chunks fed                        */
  DWORD nextOut = 0;           /* next chunk index to drain         */

  size_t chunk = ChunkSize();
  bool chunkForced = false;
  {
    char eb[32];
    DWORD el = GetEnvironmentVariableA("Z7GPU_CHUNKKB", eb, 32);
    if (el > 0 && el < 32) {
      size_t ck = (size_t)atol(eb) << 10;
      if (ck >= (64u << 10) && ck < ((size_t)1 << 22)) {
        chunk = ck;
        chunkForced = true;
      }
    }
  }
  /* adaptive window (inverse): when Z7GPU_CHUNKKB is unset we start
     at the deep 2MB window; the first drained chunk's measured
     ratio decides whether to retune DOWN to the faster 512KB path.
     Text-like data (<~6%) keeps 2MB for the whole file - worth
     ~38KB on the text bench - while mixed data pays only one 2MB
     probe chunk before dropping back. */
  if (!chunkForced) chunk = (size_t)2 << 20;
  /* mutable: the adaptive-window retune (below, inside the stream
     loop) rebuilds the session at a deeper chunk size when the first
     drained chunks prove the data highly compressible */
  size_t maxOut = ZSTD_compressBound(chunk);
  size_t seqStride = (size_t)Z7Segs() *
      ((chunk / Z7Segs() / 4 + 64) * sizeof(ZSTD_Sequence));
  DWORD nSlots = 16;
  {
    char sb[16];
    DWORD sl = GetEnvironmentVariableA("Z7GPU_SLOTS", sb, 16);
    if (sl > 0 && sl < 16) {
      int v = atoi(sb);
      if (v >= 4 && v <= 64) nSlots = (DWORD)v;
    }
  }
  size_t slotBytes = ((sizeof(Z7GpuSlotHdr) + chunk + maxOut)
      + 63) & ~(size_t)63;
  size_t mapBytes = 4096 + (size_t)nSlots * slotBytes;
  unsigned nw = _numThreads;
  if (nw > kMaxSeqWorkers) nw = kMaxSeqWorkers;
  if (nw < 1) nw = 1;

  /* adopt the process-wide pre-spawned helper when params match */
  wchar_t exe[MAX_PATH];
  DWORD inst;
  DWORD gen = 0;
  hr = E_FAIL;
  {
    Z7HostPending *pp = NULL;
    if (g_pendingSlot && InterlockedCompareExchange(
            (volatile LONG *)&g_pending, 0, 0) != 0) {
      /* claim it (only one consumer) */
      pp = (Z7HostPending *)InterlockedExchangePointer(
          (void **)&g_pending, NULL);
      g_pendingSlot = 0;
    }
    if (pp) {
      inst = pp->inst;
      wcscpy_s(exe, pp->exe);
      if (pp->th) {
        WaitForSingleObject(pp->th, Z7GPU_READY_TIMEOUT_MS);
        CloseHandle(pp->th);
      }
      if (pp->h && SUCCEEDED(pp->hr) && pp->nSlots == nSlots &&
          pp->chunk == chunk && pp->level == _Level) {
        h = *pp->h;                       /* adopt */
        hr = S_OK;
      } else if (pp->h && SUCCEEDED(pp->hr)) {
        /* params changed (-mx, CHUNKKB, -mmt): retire it */
        Z7HostCleanup(*pp->h);
      }
      if (pp->h) free(pp->h);
      free(pp);
      gen = 1;        /* gen 0 object names may still be held */
    }
    else {
      /* locate the helper next to this module */
      HMODULE self = NULL;
      inst = (DWORD)InterlockedIncrement(&g_hostInstSeq) - 1;
      if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
              (LPCWSTR)&Z7HostClose, &self) || !self)
        return CodeCpu(inStream, outStream, progress);
      DWORD n = GetModuleFileNameW(self, exe, (DWORD)MAX_PATH);
      if (!n || n >= MAX_PATH)
        return CodeCpu(inStream, outStream, progress);
      wchar_t *slash = wcsrchr(exe, L'\\');
      if (!slash) return CodeCpu(inStream, outStream, progress);
      slash[1] = 0;
      wcscat_s(exe, L"z7gpuhost.exe");
      if (GetFileAttributesW(exe) == INVALID_FILE_ATTRIBUTES)
        return CodeCpu(inStream, outStream, progress);
    }
  }

  /* spawn with a few init-time retries: the driver init crash is
     exactly what this architecture isolates, so a failed attempt
     just means "roll again" */
  while (FAILED(hr) && gen < 4) {
    hr = Z7HostSpawn(h, inst, gen, nSlots, chunk, maxOut, seqStride,
        slotBytes, mapBytes, _Level, nw, 0, exe);
    if (FAILED(hr) && dbg)
      fprintf(stderr, "[zstdgpu] host spawn gen=%lu failed\n",
          (unsigned long)gen);
    gen++;
  }
  if (FAILED(hr))
    return CodeCpu(inStream, outStream, progress);
  if (dbg) {
    fprintf(stderr,
        "[zstdgpu] host path: chunk=%u slots=%u helper up\n",
        (unsigned)chunk, (unsigned)nSlots);
    fflush(stderr);
  }

  DWORD respawns = 0;
  bool inputDone = false;
  bool eofSent = false;
  /* adaptive window (inverse): the session starts at 2MB chunks.
     once the first drained chunk's compressed size is known, a
     ratio above ~6% means the deep window is wasted - drain the
     ring, retire the session, respawn at 512KB for the faster
     parser dynamics.  text-like data keeps 2MB throughout. */
  bool adaptiveDone = chunkForced;
  unsigned adaptiveSeq = 0;
  /* the check is inlined at each nextOut++ site via this macro */
#define Z7_ADAPT_CHECK() \
  { int fired_ = 0; \
    if (!adaptiveDone && nextOut >= 1 && !inputDone) { \
      adaptiveDone = true; \
      size_t inSum_ = (size_t)nextOut * chunk; \
      if (dbg) Z7DbgLog("adapt probe: nextOut=%lu out=%zu "\
          "in=%zu\n", \
          (unsigned long)nextOut, _processedOut, inSum_); \
      if (inSum_ && _processedOut * 16 >= inSum_) { \
        while (nextOut < i) { \
          hr = Z7HostDrain(h, nextOut % nSlots, outStream, this); \
          if (FAILED(hr)) goto adapt_fail; \
          nextOut++; \
        } \
        Z7HostCleanup(h); memset(&h, 0, sizeof h); \
        chunk = (size_t)512 << 10; \
        maxOut = ZSTD_compressBound(chunk); \
        seqStride = (size_t)Z7Segs() * \
            ((chunk / Z7Segs() / 4 + 64) * sizeof(ZSTD_Sequence)); \
        slotBytes = ((sizeof(Z7GpuSlotHdr) + chunk + maxOut) + 63) \
            & ~(size_t)63; \
        mapBytes = 4096 + (size_t)nSlots * slotBytes; \
        hr = E_FAIL; \
        for (unsigned g_ = 0; g_ < 4 && FAILED(hr); g_++) \
          hr = Z7HostSpawn(h, inst, ++gen, nSlots, chunk, maxOut, \
              seqStride, slotBytes, mapBytes, _Level, nw, nextOut, \
              exe); \
        if (FAILED(hr)) goto adapt_fail; \
        adaptiveSeq++; \
        if (dbg) Z7DbgLog("adaptive: %.1f%% -> %uKB chunks\n", \
            100.0 * (double)_processedOut / (double)inSum_, \
            (unsigned)(chunk >> 10)); \
        fired_ = 1; \
      } \
    } \
    /* the slot ring was unmapped + respawned: restart the stream \
       iteration so every `sl`/event lookup sees the NEW mapping */ \
    if (fired_) continue; \
  }

  /* stream: chunk i -> slot i%nSlots; drain the slot's previous
     occupant (chunk i-nSlots) before refilling.  Helper death at
     any wait -> respawn and re-feed pending chunks. */
  if (dbg)
    Z7DbgLog("loop enter: chunk=%zu nSlots=%lu forced=%d\n",
        chunk, (unsigned long)nSlots, (int)chunkForced);
  for (;;) {
    if (!inputDone) {
      const DWORD s = i % nSlots;
      Z7GpuSlotHdr *sl = (Z7GpuSlotHdr *)(void *)
          (h.slots + (size_t)s * slotBytes);
      /* drain the slot's previous occupant (chunk i-nSlots) if it
         is still pending on the current helper generation */
      if (i >= nSlots && i - nSlots >= nextOut) {
        hr = Z7HostDrain(h, s, outStream, this);
        if (hr == Z7_E_HOSTDIED) goto respawn;
        if (FAILED(hr)) break;
        nextOut++;
        Z7_ADAPT_CHECK();
      }
      BYTE *dst = (BYTE *)(void *)(sl + 1);
      size_t got = 0;
      while (got < chunk) {
        size_t want = chunk - got;
        HRESULT rr = ReadStream(inStream, dst + got, &want);
        if (FAILED(rr)) { hr = rr; break; }
        if (want == 0) break;
        got += want;
      }
      if (FAILED(hr)) break;
      if (got == 0) { inputDone = true; continue; }
      sl->inLen = got;
      MemoryBarrier();
      SetEvent(h.evIn[s]);
      _processedIn += got;
      i++;
      if (got < chunk) inputDone = true;
      /* inverse-adaptive probe: hold the pipeline until the first
         chunk's compressed size is known - otherwise up to nSlots
         chunks could be committed to the wrong window before the
         retune check gets a sample */
      if (!adaptiveDone && i == 1 && !inputDone) {
        hr = Z7HostDrain(h, nextOut % nSlots, outStream, this);
        if (hr == Z7_E_HOSTDIED) goto respawn;
        if (FAILED(hr)) break;
        nextOut++;
        Z7_ADAPT_CHECK();
      }
      if (progress)
        if (FAILED(progress->SetRatioInfo(&_processedIn,
                &_processedOut))) { hr = E_FAIL; break; }
      continue;
    }
    if (!eofSent) {
      /* EOF marker on the next slot (drain its occupant first) */
      const DWORD s = i % nSlots;
      Z7GpuSlotHdr *sl = (Z7GpuSlotHdr *)(void *)
          (h.slots + (size_t)s * slotBytes);
      if (i >= nSlots && i - nSlots >= nextOut) {
        hr = Z7HostDrain(h, s, outStream, this);
        if (hr == Z7_E_HOSTDIED) goto respawn;
        if (FAILED(hr)) break;
        nextOut++;
        Z7_ADAPT_CHECK();
      }
      sl->inLen = 0;
      MemoryBarrier();
      SetEvent(h.evIn[s]);
      eofSent = true;
      continue;
    }
    /* drain the tail in chunk order */
    if (nextOut < i) {
      hr = Z7HostDrain(h, nextOut % nSlots, outStream, this);
      if (hr == Z7_E_HOSTDIED) goto respawn;
      if (FAILED(hr)) break;
      nextOut++;
      Z7_ADAPT_CHECK();
      continue;
    }
    break;    /* all chunks fed + drained */

respawn:
    /* helper lost: pending inputs [nextOut, i) still live in this
       generation's slots.  Spawn a fresh helper (new object names),
       copy pending inputs into its slots, replay EOF if sent.   */
    if (++respawns > 4) { hr = E_FAIL; break; }
    {
      Z7HostShm old = h;
      /* hard-stop the old helper if it is only hung */
      if (old.proc &&
          WaitForSingleObject(old.proc, 0) == WAIT_TIMEOUT)
        TerminateProcess(old.proc, 1);
      else if (old.proc)
        WaitForSingleObject(old.proc, 2000);
      hr = Z7HostSpawn(h, inst, ++gen, nSlots, chunk, maxOut,
          seqStride, slotBytes, mapBytes, _Level, nw, nextOut, exe);
      if (FAILED(hr)) { Z7HostClose(old); hr = E_FAIL; break; }
      /* replay pending chunk inputs into the new slots */
      for (DWORD c = nextOut; c < i; c++) {
        Z7GpuSlotHdr *os = (Z7GpuSlotHdr *)(void *)
            (old.slots + (size_t)(c % nSlots) * slotBytes);
        Z7GpuSlotHdr *ns = (Z7GpuSlotHdr *)(void *)
            (h.slots + (size_t)(c % nSlots) * slotBytes);
        memcpy(ns + 1, os + 1, (size_t)os->inLen);
        ns->inLen = os->inLen;
        MemoryBarrier();
        SetEvent(h.evIn[c % nSlots]);
      }
      if (eofSent) {
        Z7GpuSlotHdr *ns = (Z7GpuSlotHdr *)(void *)
            (h.slots + (size_t)(i % nSlots) * slotBytes);
        ns->inLen = 0;
        MemoryBarrier();
        SetEvent(h.evIn[i % nSlots]);
      }
      Z7HostClose(old);
      if (dbg) {
        fprintf(stderr, "[zstdgpu] host respawn gen=%lu "
            "resume at chunk %lu (fed=%lu)\n",
            (unsigned long)gen, (unsigned long)nextOut,
            (unsigned long)i);
        fflush(stderr);
      }
    }
  }

adapt_fail:
#undef Z7_ADAPT_CHECK
  (void)adaptiveSeq;
  if (dbg)
    Z7DbgLog("loop end: i=%lu nextOut=%lu in=%zu out=%zu "
        "adapt=%u\n", (unsigned long)i, (unsigned long)nextOut,
        _processedIn, _processedOut, adaptiveSeq);
  {
    DWORD err = h.ctrl ? (DWORD)h.ctrl->helperErr : 0;
    if (err && SUCCEEDED(hr)) hr = (HRESULT)err;
  }
  Z7HostCleanup(h);
  if (FAILED(hr))
    return hr;
  if (_processedIn == 0)
    return CodeCpu(inStream, outStream, progress); /* empty frame */
  return S_OK;
}

Z7_COM7F_IMF(CEncoder::Code(ISequentialInStream *inStream,
    ISequentialOutStream *outStream, const UInt64 * /*inSize*/,
    const UInt64 * /*outSize*/, ICompressProgressInfo *progress))
{
  _processedIn = 0;
  _processedOut = 0;
  /* diagnostic VEH from the 0xC0000374 investigation: debug-only —
     first-chance C++ exceptions are normal and must not be reported
     as crashes (the handler itself must never fault either) */
  static volatile LONG vehDone = 0;
  if (GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) &&
      InterlockedCompareExchange(&vehDone, 1, 0) == 0)
    AddVectoredExceptionHandler(1, ZgpuVeh);

  /* engine: seq (custom CUDA kernel via helper process, default) |
     seqi (in-process CUDA, debug only) | nvcomp | cpu */
  char engBuf[16];
  DWORD engLen = GetEnvironmentVariableA("Z7GPU_ENGINE", engBuf,
      (DWORD)sizeof(engBuf));
  const char *eng = engLen ? engBuf : "seq";
  const bool debug = GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0;

  if (strcmp(eng, "cpu") == 0)
    return CodeCpu(inStream, outStream, progress);

  /* default path: CUDA lives in z7gpuhost.exe; no driver calls here */
  if (strcmp(eng, "seq") == 0 || strcmp(eng, "host") == 0)
    return CodeGpuHostSeq(inStream, outStream, progress);

  CNvGpuDevice *gpu = CNvGpuDevice::Instance();

  if (strcmp(eng, "nvcomp") == 0) {
    gpu->Lock();
    bool ok = gpu->ProbeNvcomp() && gpu->Init();
    gpu->Unlock();
    if (!ok)
      return CodeCpu(inStream, outStream, progress);
    if (debug) {
      fprintf(stderr, "[zstdgpu] encoder: nvCOMP path, chunk=%u\n",
          (unsigned)ChunkSize());
      fflush(stderr);
    }
    return CodeGpu(inStream, outStream, progress);
  }

  /* seqi: in-process CUDA sequence path.  Debug/diagnostic only -
     on this driver build cuInit can fast-fail inside 7z.exe
     (0xC0000374); the shipped default is the out-of-process host. */
  __try {
    gpu->Lock();
    bool ok = gpu->Init();
    gpu->Unlock();
    if (!ok)
      return CodeCpu(inStream, outStream, progress);
    if (debug) {
      fprintf(stderr, "[zstdgpu] encoder: in-process seq path, "
          "chunk=%u\n", (unsigned)ChunkSize());
      fflush(stderr);
    }
    return CodeGpuSeq(inStream, outStream, progress);
  } __except(EXCEPTION_EXECUTE_HANDLER) {
    fprintf(stderr, "[zstdgpu] SEH fault 0x%08lX - %s\n",
        (unsigned long)GetExceptionCode(),
        _processedOut ? "failing" : "cpu fallback");
    fflush(stderr);
    if (_processedOut)
      return E_FAIL;
    return CodeCpu(inStream, outStream, progress);
  }
}

Z7_COM7F_IMF(CEncoder::SetNumberOfThreads(UInt32 numThreads))
{
  const UInt32 kNumThreadsMax = 256;
  if (numThreads < 1) numThreads = 1;
  if (numThreads > kNumThreadsMax) numThreads = kNumThreadsMax;
  _numThreads = numThreads;
  return S_OK;
}

}}
#endif
