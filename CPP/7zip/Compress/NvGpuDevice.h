// NvGpuDevice.h - CUDA / nvCOMP device manager for the zstd GPU codec

#ifndef __NV_GPU_DEVICE_H
#define __NV_GPU_DEVICE_H

#include <cuda_runtime.h>
#include <nvcomp/zstd.h>

#include "../../Common/MyTypes.h"

namespace NCompress {
namespace NZSTDGPU {

/*
 * Singleton holding the CUDA context + device buffers used by the
 * nvCOMP batched zstd codec. Init() tries to load nvcomp64_5.dll from
 * the directory of this module (the 7-Zip "Codecs" folder) so the
 * delay-loaded import resolves even though it is not on PATH.
 */
class CNvGpuDevice
{
public:
  static CNvGpuDevice *Instance();

  /* cheap check: a CUDA device exists (no context init) */
  bool Probe();
  /* load nvcomp64_5.dll next to this module (needed by nvCOMP paths) */
  bool ProbeNvcomp();
  /* full init incl. CUDA context/stream; call under Lock() */
  bool Init();

  void Lock()   { EnterCriticalSection(&_cs); }
  void Unlock() { LeaveCriticalSection(&_cs); }

  /*
   * Eager cleanup while the process is still healthy.  Called from
   * the encoder destructor so all CUDA work finishes before CRT
   * teardown - issuing driver calls inside process-detach races
   * ntdll heap teardown and intermittently fast-fails (0xC0000374).
   */
  void Shutdown();

  /*
   * Compress numChunks contiguous chunks of chunkBytes size each
   * (last chunk may be smaller: chunkSizes[] supplies real sizes).
   * On return, dst[i * maxOut] holds the zstd frame and
   * outSizes[i] its size.
   */
  HRESULT CompressWave(const Byte *src, const size_t *chunkSizes,
      unsigned numChunks, size_t maxOut, Byte *dst, size_t *outSizes);

  /*
   * Decompress a wave: compressed chunks reside in compWave at
   * compOffsets[]/compSizes[]; outputs go to dst at dstOffsets[]
   * with capacity dstSizes[]. Returns S_OK only if every chunk
   * decompressed; E_FAIL otherwise (caller falls back per-chunk).
   */
  HRESULT DecompressWave(const Byte *compWave,
      const size_t *compOffsets, const size_t *compSizes,
      unsigned numChunks, Byte *dst,
      const size_t *dstOffsets, const size_t *dstSizes);

  /*
   * Chunk-granular sequence-generation pipeline (no nvCOMP):
   * K ring slots, each with its own stream + buffers.  Submit()
   * enqueues H2D -> cand kernel -> pick kernel -> counts D2H;
   * PollPack() / Finish() drive the pack + packed-seq D2H; the
   * caller stitches per-segment output into ZSTD_Sequence lists.
   * Chunks are fully independent, so up to K are in flight.
   */
  HRESULT SeqPipeInit(size_t chunkBytes, unsigned segsPerChunk,
      size_t seqStride);
  unsigned SeqPipeDepth() const { return _pipeK; }
  HRESULT SeqPipeSubmit(unsigned slot, const Byte *src, size_t n);
  /* true when the counts D2H landed: pack + packed D2H enqueued,
     *totalSeqs = records the packed region holds */
  bool    SeqPipePollPack(unsigned slot, unsigned *totalSeqs);
  /* blocking: ensures pack was sent and waits for the packed D2H */
  HRESULT SeqPipeFinish(unsigned slot);
  const Byte     *SeqPipePacked(unsigned slot) const;
  const unsigned *SeqPipeCounts(unsigned slot) const;
  /* cumulative GPU-phase sums (h2d+k1, k2, d2h+pack) in ms */
  static void SeqPipeKernelMs(double out3[3]);
  static double _kms[3];
  const unsigned *SeqPipeCovs(unsigned slot) const;
  HANDLE  SeqPipeFreeEvent(unsigned slot) const;
  void    SeqPipeReset();      /* drop in-flight state (error path) */

private:
  CNvGpuDevice();
  ~CNvGpuDevice();

  HRESULT EnsureArrays(unsigned numChunks);
  HRESULT EnsureBuf(void **buf, size_t *cap, size_t need);
  void ReleaseAll();

  CRITICAL_SECTION _cs;
  void *_initMtx;   /* named mutex: serializes cuInit/context-teardown
                       across processes (driver heap race under
                       rapid process churn -> 0xC0000374 fast-fail) */
  bool _tried;
  bool _ok;
  bool _probed;
  bool _probeOk;
  HMODULE _nvcompMod;
  cudaStream_t _stream;

  /* device buffers */
  void  *_d_in;    size_t _inCap;
  void  *_d_out;   size_t _outCap;
  void  *_d_temp;  size_t _tempCap;
  void **_d_ptrsA;   /* uncompressed/in ptrs   */
  void **_d_ptrsB;   /* compressed/out ptrs    */
  size_t *_d_sizesA; /* uncompressed/in sizes  */
  size_t *_d_sizesB; /* compressed/out sizes   */
  size_t *_d_sizesC; /* out buffer capacities  */
  size_t *_d_sizesD; /* actual out sizes       */
  nvcompStatus_t *_d_status;
  unsigned _arrCap;

  /* sequence-generation pipeline: K ring slots */
  struct CSeqSlot {
    cudaStream_t st;
    cudaStream_t stP;        /* high priority: pick+pack+D2H */
    cudaEvent_t  evCounts;   /* counts D2H done   */
    cudaEvent_t  evPack;     /* packed D2H done   */
    cudaEvent_t  evStart;    /* chain head (debug timing) */
    cudaEvent_t  evK1;       /* cand kernel done */
    cudaEvent_t  evK2;       /* pick kernel done */
    Byte *d_in;
    unsigned *d_cand;
    unsigned *d_candK;         /* top-4 alt candidates per position */
    unsigned *d_choice;        /* DP parse choice scratch (4B/pos)   */
    unsigned *d_pbuf;          /* partition matcher: hist+fill+base
                                  (3*64K) then sorted[n] positions  */
    unsigned *d_hash;
    unsigned *d_cov;            /* match-interior coverage bitmap   */
    Byte *d_seqs;
    Byte *d_pack;
    unsigned *d_meta;        /* device: counts[S] + covs[S] */
    unsigned *h_meta;        /* pinned mirror of d_meta     */
    Byte *h_pack;            /* packed seqs destination     */
    Byte *d_packHost;        /* device alias of h_pack      */
    bool  hPackPinned;
    bool  packSent;
    bool  used;
    HANDLE hFree;            /* auto-reset, signaled = slot free */
    CSeqSlot() : st(NULL), stP(NULL), evCounts(NULL), evPack(NULL),
        evStart(NULL), evK1(NULL), evK2(NULL),
        d_in(NULL), d_cand(NULL), d_candK(NULL), d_choice(NULL),
        d_pbuf(NULL),
        d_hash(NULL), d_cov(NULL), d_seqs(NULL),
        d_pack(NULL), d_meta(NULL), h_meta(NULL), h_pack(NULL),
        d_packHost(NULL), hPackPinned(false), packSent(false),
        used(false),
        hFree(NULL) {}
  };
  CSeqSlot *_slots;
  unsigned _pipeK;
  size_t   _pipeChunk;
  unsigned _pipeSegs;
  size_t   _pipeSeqStride;   /* per-chunk stride (= segs*segStride) */
  unsigned _pipeHashLog;     /* hash buckets: sized vs chunk so slot
                                occupancy stays ~constant when the
                                window deepens (Z7GPU_CHUNKKB)      */
  int      _useOpt;          /* -1 unknown, 0 pick, 1 DP-opt parse  */
  int      _usePart;         /* partition-neighborhood matcher      */
  unsigned _litCost;         /* DP cost model: literal bits *256    */
  unsigned _seqBase;         /* DP cost model: seq base bits *256   */
  unsigned _repC0, _repC1, _repC2;  /* rep offset code surcharges   */
  void SeqPipeFree();
  void TeardownFree();   /* atexit-safe: host frees only, no CUDA */
  HRESULT SeqPipeSendPack(CSeqSlot &s, unsigned totalSeqs);

  /* host staging arrays (fixed capacity kMaxChunksPerWave) */
  void   **_h_ptrs;
  size_t  *_h_sizes;
  int     *_h_status;
};

}}

#endif
