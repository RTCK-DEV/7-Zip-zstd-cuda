// NvGpuDevice.cpp - CUDA / nvCOMP device manager for the zstd GPU codec

#include "StdAfx.h"

#ifndef _WIN32
#error "NvGpuDevice requires Windows"
#endif

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../Windows/System.h"

#include "NvGpuDevice.h"

namespace NCompress {
namespace NZSTDGPU {

static const unsigned kMaxChunksPerWave = 1024;

/*
 * Pin this codec module for the lifetime of the process: a pinned
 * module never receives DLL_PROCESS_DETACH and its CRT atexit list
 * never runs, so the CUDA runtime's teardown (which issues driver
 * calls) never executes inside process teardown where it can race
 * ntdll heap finalization into a 0xC0000374 fast-fail.  All real
 * cleanup is done eagerly by Shutdown() while the encoder is alive.
 */
static struct ZgpuModulePin {
  ZgpuModulePin()
  {
    HMODULE self = NULL;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_PIN,
        (LPCWSTR)this, &self);
  }
} s_modulePin;
static unsigned SeqHashLogFor(size_t chunkBytes)
{
  /* keep ~4 hash slots per input position: 4*2^hl >= 4*chunk.
     512K -> hl19 (8MB table), 1M -> 20, 2M -> 21 (128MB). */
  unsigned hl = 14;
  while (((size_t)4 << hl) < 4 * chunkBytes && hl < 22)
    hl++;
  return hl;
}

extern "C" int zgpu_cand_launch(
    const void *src, unsigned n, unsigned segsPerChunk,
    unsigned hashLog, void *hashTab, void *candOut, void *candK,
    cudaStream_t stream);

extern "C" int zgpu_opt_launch(
    const void *src, unsigned n, unsigned segsPerChunk,
    const void *candIn, const void *candKIn, void *seqOut,
    unsigned *seqCounts, unsigned *segCovs, void *choiceBuf,
    unsigned segSeqStride, unsigned litCost, unsigned seqBase,
    unsigned repC0, unsigned repC1, unsigned repC2,
    cudaStream_t stream);

extern "C" int zgpu_part_launch(
    const void *src, unsigned n, unsigned segsPerChunk,
    void *phist, void *pfill, void *pbase, void *psorted,
    void *candOut, cudaStream_t stream);

extern "C" int zgpu_pick_launch(
    const void *src, unsigned n, unsigned segsPerChunk,
    const void *candIn, const void *candKIn, void *seqOut,
    unsigned *seqCounts, unsigned *segCovs,
    unsigned segSeqStride, cudaStream_t stream);

extern "C" int zgpu_pack_launch(
    const void *seqIn, void *seqOut,
    const unsigned *seqCounts,
    unsigned nSegs, unsigned segStride, cudaStream_t stream);

CNvGpuDevice::CNvGpuDevice():
  _tried(false),
  _ok(false),
  _probed(false),
  _probeOk(false),
  _nvcompMod(NULL),
  _stream(NULL),
  _d_in(NULL),   _inCap(0),
  _d_out(NULL),  _outCap(0),
  _d_temp(NULL), _tempCap(0),
  _d_ptrsA(NULL), _d_ptrsB(NULL),
  _d_sizesA(NULL), _d_sizesB(NULL),
  _d_sizesC(NULL), _d_sizesD(NULL),
  _d_status(NULL),
  _arrCap(0),
  _slots(NULL), _pipeK(0), _pipeChunk(0), _pipeSegs(0),
  _pipeSeqStride(0), _useOpt(-1), _usePart(-1),
  _litCost(0), _seqBase(0),
  _repC0(0), _repC1(0), _repC2(0),
  _h_ptrs(NULL), _h_sizes(NULL), _h_status(NULL)
{
  InitializeCriticalSection(&_cs);
  /* cross-process serialization for the two fragile windows:
     cuInit/context-create and context-teardown.  Held only during
     Init()/Shutdown() so parallel compressions stay parallel. */
  _initMtx = CreateMutexW(NULL, FALSE, L"Local\\Z7GpuCudaInit");
}

CNvGpuDevice::~CNvGpuDevice()
{
  /* runs inside CRT/process teardown: the CUDA driver and the CRT
     heap are being dismantled in an unspecified order - issuing any
     cuda* call here can lazily init a context inside teardown and
     intermittently trigger ntdll heap fast-fail (0xC0000374).
     Do host-only cleanup; the OS reclaims device + pinned memory. */
  TeardownFree();
  if (_initMtx)
    CloseHandle((HANDLE)_initMtx);
  DeleteCriticalSection(&_cs);
}

void CNvGpuDevice::Shutdown()
{
  if (!_tried && !_slots)
    return;                          /* nothing ever initialized */
  if (_initMtx)
    WaitForSingleObject((HANDLE)_initMtx, INFINITE);
  if (_tried)
    cudaDeviceSynchronize();         /* quiesce before freeing */
  ReleaseAll();
  _tried = _ok = _probed = _probeOk = false;
  /* drop the primary context now: if it survives until CRT teardown
     the driver's release path races ntdll heap finalization and
     intermittently fast-fails (0xC0000374).  A fresh Init() lazily
     recreates it if another encoder instance is used afterwards. */
  cudaDeviceReset();
  if (_initMtx)
    ReleaseMutex((HANDLE)_initMtx);
}

void CNvGpuDevice::TeardownFree()
{
  if (!_slots)
    return;
  for (unsigned i = 0; i < _pipeK; i++) {
    CSeqSlot &s = _slots[i];
    if (s.hFree)    CloseHandle(s.hFree);
    if (s.h_pack && !s.hPackPinned)
      free(s.h_pack);
  }
  free(_slots);
  _slots = NULL;
  _pipeK = 0;
}

CNvGpuDevice *CNvGpuDevice::Instance()
{
  static CNvGpuDevice g_inst;
  return &g_inst;
}

void CNvGpuDevice::ReleaseAll()
{
  if (_d_in)     { cudaFree(_d_in);     _d_in = NULL; }
  if (_d_out)    { cudaFree(_d_out);    _d_out = NULL; }
  if (_d_temp)   { cudaFree(_d_temp);   _d_temp = NULL; }
  if (_d_ptrsA)  { cudaFree(_d_ptrsA);  _d_ptrsA = NULL; }
  if (_d_ptrsB)  { cudaFree(_d_ptrsB);  _d_ptrsB = NULL; }
  if (_d_sizesA) { cudaFree(_d_sizesA); _d_sizesA = NULL; }
  if (_d_sizesB) { cudaFree(_d_sizesB); _d_sizesB = NULL; }
  if (_d_sizesC) { cudaFree(_d_sizesC); _d_sizesC = NULL; }
  if (_d_sizesD) { cudaFree(_d_sizesD); _d_sizesD = NULL; }
  if (_d_status) { cudaFree(_d_status); _d_status = NULL; }
  if (_stream)   { cudaStreamDestroy(_stream); _stream = NULL; }
  if (_h_ptrs)   { cudaFreeHost(_h_ptrs);   _h_ptrs = NULL; }
  if (_h_sizes)  { cudaFreeHost(_h_sizes);  _h_sizes = NULL; }
  if (_h_status) { cudaFreeHost(_h_status); _h_status = NULL; }
  SeqPipeFree();
  _inCap = _outCap = _tempCap = _arrCap = 0;
}

/*
 * Load a sibling DLL that sits next to this codec module (the 7-Zip
 * "Codecs" folder is not on the DLL search path).  Both nvcomp and
 * the CUDA runtime are delay-loaded; preloading them explicitly makes
 * the delay-load binder resolve to the already-loaded module.
 */
static HMODULE LoadSibling(const wchar_t *name)
{
  static int s_anchor;
  HMODULE self = NULL;
  wchar_t path[MAX_PATH * 2];
  HMODULE mod = NULL;
  if (GetModuleHandleExW(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCWSTR)&s_anchor, &self) && self)
  {
    DWORD n = GetModuleFileNameW(self, path, (DWORD)Z7_ARRAY_SIZE(path));
    if (n && n < Z7_ARRAY_SIZE(path))
    {
      wchar_t *slash = wcsrchr(path, L'\\');
      if (slash)
      {
        size_t remain = Z7_ARRAY_SIZE(path) - (size_t)(slash + 1 - path);
        wcscpy_s(slash + 1, remain, name);
        mod = LoadLibraryW(path);
      }
    }
  }
  if (!mod)
    mod = LoadLibraryW(name);          /* PATH fallback */
  return mod;
}

bool CNvGpuDevice::Probe()
{
  if (_probed)
    return _probeOk;
  _probed = true;
  int devCount = 0;
  _probeOk =
      (cudaGetDeviceCount(&devCount) == cudaSuccess && devCount > 0);
  return _probeOk;
}

bool CNvGpuDevice::ProbeNvcomp()
{
  if (_nvcompMod)
    return true;
  _nvcompMod = LoadSibling(L"nvcomp64_5.dll");
  return _nvcompMod != NULL;
}

bool CNvGpuDevice::Init()
{
  if (_tried)
    return _ok;
  _tried = true;
  if (_initMtx)
    WaitForSingleObject((HANDLE)_initMtx, INFINITE);
  bool ok = Probe();
  if (ok && cudaStreamCreate(&_stream) != cudaSuccess)
    ok = false;

  /* pinned host staging: async copies only stream at full rate
     when the host side is page-locked */
  if (ok &&
      (cudaHostAlloc((void **)&_h_ptrs,
          kMaxChunksPerWave * sizeof(void *),
          cudaHostAllocDefault) != cudaSuccess ||
      cudaHostAlloc((void **)&_h_sizes,
          kMaxChunksPerWave * sizeof(size_t),
          cudaHostAllocDefault) != cudaSuccess ||
      cudaHostAlloc((void **)&_h_status,
          kMaxChunksPerWave * sizeof(int),
          cudaHostAllocDefault) != cudaSuccess))
    ok = false;
  if (_initMtx)
    ReleaseMutex((HANDLE)_initMtx);

  _ok = ok;
  return ok;
}

HRESULT CNvGpuDevice::EnsureBuf(void **buf, size_t *cap, size_t need)
{
  if (*cap >= need)
    return S_OK;
  if (*buf)
    cudaFree(*buf);
  *buf = NULL;
  *cap = 0;
  if (cudaMalloc(buf, need) != cudaSuccess)
    return E_OUTOFMEMORY;
  *cap = need;
  return S_OK;
}

HRESULT CNvGpuDevice::EnsureArrays(unsigned numChunks)
{
  if (numChunks > kMaxChunksPerWave)
    return E_INVALIDARG;
  if (_arrCap >= numChunks)
    return S_OK;
  if (_d_ptrsA)  { cudaFree(_d_ptrsA);  _d_ptrsA = NULL; }
  if (_d_ptrsB)  { cudaFree(_d_ptrsB);  _d_ptrsB = NULL; }
  if (_d_sizesA) { cudaFree(_d_sizesA); _d_sizesA = NULL; }
  if (_d_sizesB) { cudaFree(_d_sizesB); _d_sizesB = NULL; }
  if (_d_sizesC) { cudaFree(_d_sizesC); _d_sizesC = NULL; }
  if (_d_sizesD) { cudaFree(_d_sizesD); _d_sizesD = NULL; }
  if (_d_status) { cudaFree(_d_status); _d_status = NULL; }
  _arrCap = 0;
  if (cudaMalloc((void **)&_d_ptrsA,  numChunks * sizeof(void *))  != cudaSuccess ||
      cudaMalloc((void **)&_d_ptrsB,  numChunks * sizeof(void *))  != cudaSuccess ||
      cudaMalloc((void **)&_d_sizesA, numChunks * sizeof(size_t))  != cudaSuccess ||
      cudaMalloc((void **)&_d_sizesB, numChunks * sizeof(size_t))  != cudaSuccess ||
      cudaMalloc((void **)&_d_sizesC, numChunks * sizeof(size_t))  != cudaSuccess ||
      cudaMalloc((void **)&_d_sizesD, numChunks * sizeof(size_t))  != cudaSuccess ||
      cudaMalloc((void **)&_d_status, numChunks * sizeof(nvcompStatus_t)) != cudaSuccess)
    return E_OUTOFMEMORY;
  _arrCap = numChunks;
  return S_OK;
}

HRESULT CNvGpuDevice::CompressWave(const Byte *src, const size_t *chunkSizes,
    unsigned numChunks, size_t maxOut, Byte *dst, size_t *outSizes)
{
  if (!ProbeNvcomp())
    return E_FAIL;

  size_t inTotal = 0;
  size_t maxIn = 0;
  for (unsigned i = 0; i < numChunks; i++)
  {
    inTotal += chunkSizes[i];
    if (chunkSizes[i] > maxIn)
      maxIn = chunkSizes[i];
  }

  RINOK(EnsureArrays(numChunks))
  RINOK(EnsureBuf(&_d_in,  &_inCap,  inTotal))
  RINOK(EnsureBuf(&_d_out, &_outCap, maxOut * numChunks))

  if (cudaMemcpy(_d_in, src, inTotal, cudaMemcpyHostToDevice) != cudaSuccess)
    return E_FAIL;

  {
    Byte *base = (Byte *)_d_in;
    for (unsigned i = 0; i < numChunks; i++)
    {
      _h_ptrs[i] = base;
      base += chunkSizes[i];
    }
    if (cudaMemcpy(_d_ptrsA, _h_ptrs, numChunks * sizeof(void *),
            cudaMemcpyHostToDevice) != cudaSuccess)
      return E_FAIL;
    if (cudaMemcpy(_d_sizesA, chunkSizes, numChunks * sizeof(size_t),
            cudaMemcpyHostToDevice) != cudaSuccess)
      return E_FAIL;
  }
  {
    Byte *base = (Byte *)_d_out;
    for (unsigned i = 0; i < numChunks; i++)
    {
      _h_ptrs[i] = base;
      base += maxOut;
    }
    if (cudaMemcpy(_d_ptrsB, _h_ptrs, numChunks * sizeof(void *),
            cudaMemcpyHostToDevice) != cudaSuccess)
      return E_FAIL;
  }

  size_t tempBytes = 0;
  if (nvcompBatchedZstdCompressGetTempSizeAsync(numChunks, maxIn,
          nvcompBatchedZstdCompressDefaultOpts, &tempBytes, inTotal) != nvcompSuccess)
    return E_FAIL;
  RINOK(EnsureBuf(&_d_temp, &_tempCap, tempBytes))

  if (nvcompBatchedZstdCompressAsync(
          (const void *const *)_d_ptrsA, _d_sizesA, maxIn, numChunks,
          _d_temp, tempBytes,
          _d_ptrsB, _d_sizesB,
          nvcompBatchedZstdCompressDefaultOpts, _d_status, _stream) != nvcompSuccess)
    return E_FAIL;

  if (cudaStreamSynchronize(_stream) != cudaSuccess)
    return E_FAIL;

  {
    if (cudaMemcpy(_h_status, _d_status, numChunks * sizeof(nvcompStatus_t),
            cudaMemcpyDeviceToHost) != cudaSuccess)
      return E_FAIL;
    for (unsigned i = 0; i < numChunks; i++)
      if (_h_status[i] != nvcompSuccess)
        return E_FAIL;
  }
  if (cudaMemcpy(outSizes, _d_sizesB, numChunks * sizeof(size_t),
          cudaMemcpyDeviceToHost) != cudaSuccess)
    return E_FAIL;

  for (unsigned i = 0; i < numChunks; i++)
  {
    if (cudaMemcpy(dst + i * maxOut, (Byte *)_d_out + i * maxOut,
            outSizes[i], cudaMemcpyDeviceToHost) != cudaSuccess)
      return E_FAIL;
  }
  return S_OK;
}

HRESULT CNvGpuDevice::DecompressWave(const Byte *compWave,
    const size_t *compOffsets, const size_t *compSizes,
    unsigned numChunks, Byte *dst,
    const size_t *dstOffsets, const size_t *dstSizes)
{
  if (!ProbeNvcomp())
    return E_FAIL;

  size_t compTotal = 0;
  size_t maxComp = 0;
  size_t maxOut = 0;
  size_t outTotal = 0;
  for (unsigned i = 0; i < numChunks; i++)
  {
    size_t end = compOffsets[i] + compSizes[i];
    if (end > compTotal) compTotal = end;
    if (compSizes[i] > maxComp) maxComp = compSizes[i];
    if (dstSizes[i] > maxOut) maxOut = dstSizes[i];
    outTotal += dstSizes[i];
  }

  RINOK(EnsureArrays(numChunks))
  RINOK(EnsureBuf(&_d_in,  &_inCap,  compTotal))
  RINOK(EnsureBuf(&_d_out, &_outCap, outTotal))

  if (cudaMemcpy(_d_in, compWave, compTotal, cudaMemcpyHostToDevice) != cudaSuccess)
    return E_FAIL;

  {
    Byte *inBase = (Byte *)_d_in;
    for (unsigned i = 0; i < numChunks; i++)
      _h_ptrs[i] = inBase + compOffsets[i];
    if (cudaMemcpy(_d_ptrsA, _h_ptrs, numChunks * sizeof(void *),
            cudaMemcpyHostToDevice) != cudaSuccess)
      return E_FAIL;
    if (cudaMemcpy(_d_sizesA, compSizes, numChunks * sizeof(size_t),
            cudaMemcpyHostToDevice) != cudaSuccess)
      return E_FAIL;
  }
  {
    Byte *outBase = (Byte *)_d_out;
    for (unsigned i = 0; i < numChunks; i++)
      _h_ptrs[i] = outBase + dstOffsets[i];
    if (cudaMemcpy(_d_ptrsB, _h_ptrs, numChunks * sizeof(void *),
            cudaMemcpyHostToDevice) != cudaSuccess)
      return E_FAIL;
    if (cudaMemcpy(_d_sizesB, dstSizes, numChunks * sizeof(size_t),
            cudaMemcpyHostToDevice) != cudaSuccess)
      return E_FAIL;
  }

  size_t tempBytes = 0;
  nvcompBatchedZstdDecompressOpts_t dopts = nvcompBatchedZstdDecompressDefaultOpts;
  if (nvcompBatchedZstdDecompressGetTempSizeAsync(numChunks, maxOut,
          dopts, &tempBytes, outTotal) != nvcompSuccess)
    return E_FAIL;
  RINOK(EnsureBuf(&_d_temp, &_tempCap, tempBytes))

  if (nvcompBatchedZstdDecompressAsync(
          (const void *const *)_d_ptrsA, _d_sizesA,
          _d_sizesB /* buffer sizes */, _d_sizesC /* actual out sizes */,
          numChunks, _d_temp, tempBytes,
          _d_ptrsB, dopts, _d_status, _stream) != nvcompSuccess)
    return E_FAIL;

  if (cudaStreamSynchronize(_stream) != cudaSuccess)
    return E_FAIL;

  {
    if (cudaMemcpy(_h_status, _d_status, numChunks * sizeof(nvcompStatus_t),
            cudaMemcpyDeviceToHost) != cudaSuccess)
      return E_FAIL;
    for (unsigned i = 0; i < numChunks; i++)
      if (_h_status[i] != nvcompSuccess)
        return E_FAIL;
  }
  if (cudaMemcpy(_h_sizes, _d_sizesC, numChunks * sizeof(size_t),
          cudaMemcpyDeviceToHost) != cudaSuccess)
    return E_FAIL;

  for (unsigned i = 0; i < numChunks; i++)
  {
    if (_h_sizes[i] > dstSizes[i])
      return E_FAIL;
    if (cudaMemcpy(dst + dstOffsets[i], (Byte *)_d_out + dstOffsets[i],
            _h_sizes[i], cudaMemcpyDeviceToHost) != cudaSuccess)
      return E_FAIL;
  }
  return S_OK;
}

/*
 * Chunk-granular sequence-generation pipeline.  Each ring slot owns a
 * stream plus private device/host buffers sized for one chunk, so up
 * to K chunks are in flight: chunk i's H2D/kernels overlap chunk j's
 * pack D2H and chunk k's host stitch+encode.  The hash table is never
 * cleared - stale positions are revalidated by byte comparison in the
 * candidate kernel, so correctness only depends on c < p.
 */
void CNvGpuDevice::SeqPipeFree()
{
  if (!_slots)
    return;
  for (unsigned i = 0; i < _pipeK; i++) {
    CSeqSlot &s = _slots[i];
    if (s.d_in)    cudaFree(s.d_in);
    if (s.d_cand)  cudaFree(s.d_cand);
    if (s.d_candK) cudaFree(s.d_candK);
    if (s.d_pbuf)  cudaFree(s.d_pbuf);
    if (s.d_choice) cudaFree(s.d_choice);
    if (s.d_hash)  cudaFree(s.d_hash);
    if (s.d_cov)   cudaFree(s.d_cov);
    if (s.d_seqs)  cudaFree(s.d_seqs);
    if (s.d_pack)  cudaFree(s.d_pack);
    if (s.d_meta)  cudaFree(s.d_meta);
    if (s.h_meta)  cudaFreeHost(s.h_meta);
    if (s.h_pack) {
      if (s.hPackPinned) cudaFreeHost(s.h_pack);
      else free(s.h_pack);
    }
    if (s.evCounts) cudaEventDestroy(s.evCounts);
    if (s.evPack)   cudaEventDestroy(s.evPack);
    if (s.evStart)  cudaEventDestroy(s.evStart);
    if (s.evK1)     cudaEventDestroy(s.evK1);
    if (s.evK2)     cudaEventDestroy(s.evK2);
    if (s.st)       cudaStreamDestroy(s.st);
    if (s.stP)      cudaStreamDestroy(s.stP);
    if (s.hFree)    CloseHandle(s.hFree);
  }
  free(_slots);
  _slots = NULL;
  _pipeK = 0;
  _pipeChunk = _pipeSeqStride = 0;
  _pipeSegs = 0;
}

HRESULT CNvGpuDevice::SeqPipeInit(size_t chunkBytes,
    unsigned segsPerChunk, size_t seqStride)
{
  if (segsPerChunk == 0 || segsPerChunk > 1024 ||
      seqStride % (segsPerChunk * sizeof(unsigned)) != 0 ||
      chunkBytes >= ((size_t)1 << 22)) /* cand kernel packs pos in
      22 slot bits (tag takes the top 10) */
    return E_INVALIDARG;
  if (_slots && _pipeChunk == chunkBytes &&
      _pipeSegs == segsPerChunk && _pipeSeqStride == seqStride)
    return S_OK;

  SeqPipeFree();

  /* deeper ring for small chunks; cap device+host footprint */
  unsigned k = 24;
  if (chunkBytes > (size_t)2 << 20) k = 6;
  if (chunkBytes > (size_t)8 << 20) k = 3;

  if (_useOpt < 0) {
    /* default: classic greedy+lazy pick kernel - the DP optimal
       parse (Z7GPU_OPT=1) loses on both speed and ratio, kept for
       research only.  Costs in 1/256-bit units. */
    char buf[32];
    DWORD l = GetEnvironmentVariableA("Z7GPU_OPT", buf, sizeof buf);
    _useOpt = (l && l < sizeof buf && buf[0] == '1') ? 1 : 0;
    l = GetEnvironmentVariableA("Z7GPU_LITC", buf, sizeof buf);
    _litCost = (l && l < sizeof buf) ? (unsigned)atoi(buf) : 3072;
    l = GetEnvironmentVariableA("Z7GPU_SEQB", buf, sizeof buf);
    _seqBase = (l && l < sizeof buf) ? (unsigned)atoi(buf) : 2304;
    l = GetEnvironmentVariableA("Z7GPU_REPC", buf, sizeof buf);
    _repC0 = _repC1 = _repC2 =
        (l && l < sizeof buf) ? (unsigned)atoi(buf) : 768;
    l = GetEnvironmentVariableA("Z7GPU_REPC1", buf, sizeof buf);
    if (l && l < sizeof buf) _repC1 = (unsigned)atoi(buf);
    l = GetEnvironmentVariableA("Z7GPU_REPC2", buf, sizeof buf);
    if (l && l < sizeof buf) _repC2 = (unsigned)atoi(buf);
    /* partition-neighborhood matcher: additive deep-match path.
       Measured ~ratio-neutral on the corpus (+~160ms wall for the
       extra kernels) - opt-in research path via Z7GPU_PART=1. */
    l = GetEnvironmentVariableA("Z7GPU_PART", buf, sizeof buf);
    _usePart = (l && l < sizeof buf && buf[0] == '1') ? 1 : 0;
  }

  _slots = (CSeqSlot *)calloc(k, sizeof(CSeqSlot));
  if (!_slots)
    return E_OUTOFMEMORY;

  _pipeHashLog = SeqHashLogFor(chunkBytes);
  const size_t packCap = seqStride;   /* worst case: all slots full */
  const size_t hashBytes = ((size_t)1 << (_pipeHashLog + 2))
      * sizeof(unsigned);
  const size_t metaBytes = 2 * segsPerChunk * sizeof(unsigned);

  /* pick/pack work gets its own high-priority stream per slot: the
     128-thread pick blocks would otherwise starve behind the
     multi-thousand-block candidate kernels of other slots */
  int prioLo = 0, prioHi = 0;
  if (cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi)
          != cudaSuccess)
    (void)cudaGetLastError();

  for (unsigned i = 0; i < k; i++) {
    CSeqSlot &s = _slots[i];
    if (cudaStreamCreateWithFlags(&s.st, cudaStreamNonBlocking)
            != cudaSuccess ||
        cudaStreamCreateWithPriority(&s.stP, cudaStreamNonBlocking,
            prioHi) != cudaSuccess ||
        cudaEventCreateWithFlags(&s.evCounts, cudaEventDisableTiming)
            != cudaSuccess ||
        cudaEventCreateWithFlags(&s.evPack, 0)
            != cudaSuccess ||
        cudaEventCreateWithFlags(&s.evStart, 0) != cudaSuccess ||
        cudaEventCreateWithFlags(&s.evK1, 0) != cudaSuccess ||
        cudaEventCreateWithFlags(&s.evK2, 0) != cudaSuccess ||
        cudaMalloc((void **)&s.d_in, chunkBytes) != cudaSuccess ||
        cudaMalloc((void **)&s.d_cand,
            chunkBytes * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc((void **)&s.d_candK,
            chunkBytes * 4 * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc((void **)&s.d_choice,
            (chunkBytes + 1) * sizeof(unsigned)) != cudaSuccess ||
        (_usePart && cudaMalloc((void **)&s.d_pbuf,
            ((size_t)3 * 65536 + chunkBytes) * sizeof(unsigned))
            != cudaSuccess) ||
        cudaMalloc((void **)&s.d_hash, hashBytes) != cudaSuccess ||
        cudaMalloc((void **)&s.d_cov,
            (chunkBytes / 32 + 1) * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc((void **)&s.d_seqs, packCap) != cudaSuccess ||
        cudaMalloc((void **)&s.d_pack, packCap) != cudaSuccess ||
        cudaMalloc((void **)&s.d_meta, metaBytes) != cudaSuccess ||
        cudaHostAlloc((void **)&s.h_meta, metaBytes,
            cudaHostAllocDefault) != cudaSuccess)
      goto fail;
    if (cudaHostAlloc((void **)&s.h_pack, packCap,
            cudaHostAllocMapped) == cudaSuccess &&
        cudaHostGetDevicePointer((void **)&s.d_packHost,
            s.h_pack, 0) == cudaSuccess)
      s.hPackPinned = true;
    else {
      (void)cudaGetLastError();
      if (s.h_pack)
        cudaFreeHost(s.h_pack);
      s.h_pack = (Byte *)malloc(packCap);
      s.hPackPinned = false;
      if (!s.h_pack)
        goto fail;
    }
    /* auto-reset, initially signaled: the slot starts free */
    s.hFree = CreateEventW(NULL, FALSE, TRUE, NULL);
    if (!s.hFree)
      goto fail;
  }

  _pipeK = k;
  _pipeChunk = chunkBytes;
  _pipeSegs = segsPerChunk;
  _pipeSeqStride = seqStride;
  return S_OK;

fail:
  SeqPipeFree();
  return E_OUTOFMEMORY;
}

void CNvGpuDevice::SeqPipeReset()
{
  for (unsigned i = 0; i < _pipeK; i++) {
    _slots[i].used = false;
    _slots[i].packSent = false;
    if (_slots[i].hFree)
      SetEvent(_slots[i].hFree);      /* recover the free signal */
  }
}

HANDLE CNvGpuDevice::SeqPipeFreeEvent(unsigned slot) const
{
  return _slots[slot].hFree;
}

HRESULT CNvGpuDevice::SeqPipeSubmit(unsigned slot, const Byte *src,
    size_t n)
{
  if (!_slots || slot >= _pipeK || n > _pipeChunk)
    return E_INVALIDARG;
  CSeqSlot &s = _slots[slot];
  if (s.used)
    return E_FAIL;                  /* caller must Finish() first */

  const unsigned segStride = (unsigned)(_pipeSeqStride / _pipeSegs);

#define ZGPU_CHK(tag, call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
      fprintf(stderr, "[zstdgpu]   %s -> %d %s\n", tag, (int)e_, \
          cudaGetErrorString(e_)); \
      return E_FAIL; \
    } \
  } while (0)

  ZGPU_CHK("evs", cudaEventRecord(s.evStart, s.st));
  ZGPU_CHK("h2d", cudaMemcpyAsync(s.d_in, src, n,
      cudaMemcpyHostToDevice, s.st));
  /* clear the hash table: stale positions from the previous chunk on
     this slot are byte-verified anyway (safe), but they displace
     fresh entries and measurably hurt the ratio */
  /* hash table clear is unnecessary: stale slots from the previous
     chunk are filtered twice - cc<p rejects forward refs, and the
     rd64 verify in candLen64 rejects tag-collision survivors, so
     skipping the 4MB memset only costs a rare extra eval load */
  if (0) ZGPU_CHK("hash-clr", cudaMemsetAsync(s.d_hash, 0xFF,
      ((size_t)1 << (_pipeHashLog + 2)) * sizeof(unsigned), s.st));
  /* cand table merges atomicMax offers -> must start zeroed */
  ZGPU_CHK("cand-clr", cudaMemsetAsync(s.d_cand, 0,
      n * sizeof(unsigned), s.st));
  /* candK rows are always generated now: the pick kernel consumes
     them for rep-aware best-of-5 selection, so the KMODE=true k1
     template runs on both paths */
  if (zgpu_cand_launch(s.d_in, (unsigned)n, _pipeSegs, _pipeHashLog,
          s.d_hash, s.d_cand, s.d_candK,
          s.st) != cudaSuccess)
    return E_FAIL;
  if (_usePart) {
    /* partition matcher: hist/fill/base (3*64K u32) then sorted[n].
       phist must be zeroed; pfill is cleared by part_launch after
       the scan consumed it. */
    unsigned *ph = s.d_pbuf, *pf = ph + 65536, *pb = ph + 131072,
             *ps = ph + 196608;
    ZGPU_CHK("phist-clr", cudaMemsetAsync(ph, 0,
        65536 * sizeof(unsigned), s.st));
    if (zgpu_part_launch(s.d_in, (unsigned)n, _pipeSegs, ph, pf, pb,
            ps, s.d_cand, s.st) != cudaSuccess)
      return E_FAIL;
  }
  ZGPU_CHK("ev1", cudaEventRecord(s.evK1, s.st));
  /* pick kernel on the high-priority stream, gated by evK1 */
  ZGPU_CHK("pw", cudaStreamWaitEvent(s.stP, s.evK1, 0));
  if (_useOpt) {
    if (zgpu_opt_launch(s.d_in, (unsigned)n, _pipeSegs, s.d_cand,
            s.d_candK, s.d_seqs, s.d_meta, s.d_meta + _pipeSegs,
            s.d_choice, segStride, _litCost, _seqBase,
            _repC0, _repC1, _repC2, s.stP) != cudaSuccess)
      return E_FAIL;
  } else if (zgpu_pick_launch(s.d_in, (unsigned)n, _pipeSegs, s.d_cand,
          s.d_candK, s.d_seqs, s.d_meta, s.d_meta + _pipeSegs,
          segStride, s.stP) != cudaSuccess)
    return E_FAIL;
  ZGPU_CHK("ev2", cudaEventRecord(s.evK2, s.stP));
  ZGPU_CHK("cnt-d2h", cudaMemcpyAsync(s.h_meta, s.d_meta,
      2 * _pipeSegs * sizeof(unsigned), cudaMemcpyDeviceToHost, s.stP));
  ZGPU_CHK("evc", cudaEventRecord(s.evCounts, s.stP));
  /* when h_pack is mapped the pack kernel writes straight into
     pinned host memory: evPack then covers the whole chain and no
     sized D2H round-trip is needed.  Otherwise pack into device
     memory and let SeqPipeSendPack enqueue the copy once the real
     record count is known. */
  if (s.hPackPinned) {
    if (zgpu_pack_launch(s.d_seqs, s.d_packHost, s.d_meta,
            _pipeSegs, segStride, s.stP) != cudaSuccess)
      return E_FAIL;
    ZGPU_CHK("evp", cudaEventRecord(s.evPack, s.stP));
    s.packSent = true;
  } else {
    if (zgpu_pack_launch(s.d_seqs, s.d_pack, s.d_meta, _pipeSegs,
            segStride, s.stP) != cudaSuccess)
      return E_FAIL;
    s.packSent = false;
  }
#undef ZGPU_CHK

  s.used = true;
  if (GetEnvironmentVariableA("Z7GPU_SYNC2", NULL, 0)) {
    cudaError_t e1 = cudaStreamSynchronize(s.stP);
    unsigned dbg[8] = {0};
    cudaMemcpy(dbg, s.d_meta, sizeof dbg,
        cudaMemcpyDeviceToHost);
    cudaError_t e2 = cudaGetLastError();
    fprintf(stderr, "[zstdgpu] syncall slot=%u e1=%d e2=%d "
        "m0=%u m1=%u hc0=%u hc1=%u\n", slot, (int)e1, (int)e2,
        dbg[0], dbg[1], s.h_meta[0], s.h_meta[1]);
  }
  return S_OK;
}

/* the pack kernel was enqueued at submit; once counts are known we
   only need the right-sized D2H of the packed region */
HRESULT CNvGpuDevice::SeqPipeSendPack(CSeqSlot &s, unsigned totalSeqs)
{
  if (totalSeqs &&
      cudaMemcpyAsync(s.h_pack, s.d_pack, (size_t)totalSeqs * 16,
          cudaMemcpyDeviceToHost, s.stP) != cudaSuccess)
    return E_FAIL;
  if (cudaEventRecord(s.evPack, s.stP) != cudaSuccess)
    return E_FAIL;
  s.packSent = true;
  return S_OK;
}

bool CNvGpuDevice::SeqPipePollPack(unsigned slot, unsigned *totalSeqs)
{
  CSeqSlot &s = _slots[slot];
  if (!s.used || s.packSent)
    return true;
  if (cudaEventQuery(s.evCounts) != cudaSuccess)
    return false;
  unsigned tot = 0;
  for (unsigned i = 0; i < _pipeSegs; i++)
    tot += s.h_meta[i];
  *totalSeqs = tot;
  return SUCCEEDED(SeqPipeSendPack(s, tot));
}

HRESULT CNvGpuDevice::SeqPipeFinish(unsigned slot)
{
  if (!_slots || slot >= _pipeK)
    return E_INVALIDARG;
  CSeqSlot &s = _slots[slot];
  if (!s.used)
    return S_OK;
  if (!s.packSent) {
    if (cudaEventSynchronize(s.evCounts) != cudaSuccess)
      return E_FAIL;
    unsigned tot = 0;
    for (unsigned i = 0; i < _pipeSegs; i++)
      tot += s.h_meta[i];
    RINOK(SeqPipeSendPack(s, tot))
  }
  cudaError_t eps = cudaEventSynchronize(s.evPack);
  if (GetEnvironmentVariableA("Z7GPU_SYNCALL", NULL, 0))
    fprintf(stderr, "[zstdgpu] fin slot=%u eps=%d hm0=%u hm1=%u\n",
        slot, (int)eps, s.h_meta[0], s.h_meta[1]);
  if (eps != cudaSuccess)
    return E_FAIL;
  if (GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0)) {
    static unsigned dbgN;
    if ((dbgN++ & 31) == 0) {
      unsigned tot = 0;
      for (unsigned i = 0; i < _pipeSegs; i++)
        tot += s.h_meta[i];
      fprintf(stderr, "[zstdgpu]   seqs=%u covs=%u\n", tot,
          s.h_meta[_pipeSegs]);
    }
    static unsigned dbgM;
    float ms0 = 0, ms1 = 0, ms2 = 0;
    cudaEventElapsedTime(&ms0, s.evStart, s.evK1);
    cudaEventElapsedTime(&ms1, s.evK1, s.evK2);
    cudaEventElapsedTime(&ms2, s.evK2, s.evPack);
    _kms[0] += ms0; _kms[1] += ms1; _kms[2] += ms2;
    if ((dbgM++ & 31) == 0)
      fprintf(stderr, "[zstdgpu]   h2d+k1=%.2f k2=%.2f d2h+pack=%.2f\n",
          (double)ms0, (double)ms1, (double)ms2);
  }
  s.used = false;
  return S_OK;
}

double CNvGpuDevice::_kms[3];

void CNvGpuDevice::SeqPipeKernelMs(double out3[3])
{
  out3[0] = _kms[0]; out3[1] = _kms[1]; out3[2] = _kms[2];
}

const Byte *CNvGpuDevice::SeqPipePacked(unsigned slot) const
{
  return _slots[slot].h_pack;
}

const unsigned *CNvGpuDevice::SeqPipeCounts(unsigned slot) const
{
  return _slots[slot].h_meta;
}

const unsigned *CNvGpuDevice::SeqPipeCovs(unsigned slot) const
{
  return _slots[slot].h_meta + _pipeSegs;
}

}}
