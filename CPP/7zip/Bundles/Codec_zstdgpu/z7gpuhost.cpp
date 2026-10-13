/* z7gpuhost.cpp - out-of-process CUDA zstd sequence compressor.

   Why this exists: cuInit/nvcuda64 intermittently fast-fails with
   STATUS_HEAP_CORRUPTION (0xC0000374, HEAP_FAILURE_INVALID_ENTRY)
   inside nested module loads on driver 616.92.  The fault is a
   process-fatal fast-fail no handler can ride out, so all CUDA work
   lives here instead of in 7z.exe.  If this process dies the codec
   just falls back to CPU zstd - the archive stays valid.

   Protocol: parent creates a pagefile-backed mapping + named events
   (see GpuHostShared.h), fills Z7GpuHostCtrl, spawns us, and streams
   raw chunks through the slot ring; we reply with zstd frames.
*/

#include <windows.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include "../../Compress/GpuHostShared.h"
#include "../../Compress/NvGpuDevice.h"

using namespace NCompress::NZSTDGPU;

/* ---- copied from ZstdGpuEncoder.cpp: fold per-segment GPU output
        into a single ZSTD_Sequence list -------------------------- */
static unsigned StitchSegs(const BYTE *pack, const unsigned *counts,
    const unsigned *covs, size_t n, unsigned segs, BYTE *dst)
{
  unsigned wpos = 0;
  unsigned pending = 0;
  unsigned packOff = 0;
  const size_t segLen = (n + segs - 1) / segs;
  for (unsigned s = 0; s < segs; s++) {
    const unsigned cnt = counts[s];
    const size_t lo = (size_t)s * segLen;
    const size_t hi = lo + segLen < n ? lo + segLen : n;
    const BYTE *raw = pack + (size_t)packOff * 16;
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

/* ------------------------------------------------------------ */

struct ShmView {
  HANDLE map;
  Z7GpuHostCtrl *ctrl;
  BYTE *slots;            /* ctrl + page-aligned slot array        */
  HANDLE evReady, evQuit;
  HANDLE *evIn, *evOut;
  DWORD nSlots;
};

static Z7GpuSlotHdr *SlotHdr(ShmView &v, unsigned s)
{
  return (Z7GpuSlotHdr *)(void *)(v.slots + (size_t)s * v.ctrl->slotBytes);
}
static BYTE *SlotIn(ShmView &v, unsigned s)
{
  return (BYTE *)(void *)(v.slots + (size_t)s * v.ctrl->slotBytes
      + sizeof(Z7GpuSlotHdr));
}
static BYTE *SlotOut(ShmView &v, unsigned s)
{
  return SlotIn(v, s) + v.ctrl->chunk;
}

struct EncShared {
  HANDLE workSem;           /* +1 per queued chunk                   */
  volatile LONG next;       /* dequeue index shared by all workers   */
  volatile LONG quit;       /* set on EOF                            */
  volatile LONG err;        /* first helper-side failure             */
  volatile LONG done;       /* tokens fully processed                */
  struct Tok { DWORD shmSlot; DWORD pipeSlot; DWORD inLen; } *toks;
};

struct EncJob {
  CNvGpuDevice *gpu;
  unsigned pipeK;
  unsigned segs;
  ZSTD_CCtx *ctx;
  ZSTD_CCtx *ctxCpu;        /* plain zstd ctx for per-chunk min()   */
  BYTE *cpuScratch;         /* candidate output (maxOut)            */
  BYTE *seqBuf;             /* per-worker stitched seq scratch      */
  size_t seqStride;
  size_t chunk;
  size_t maxOut;
  int level;
  ShmView *v;
  EncShared *sh;
  unsigned long long finMs;  /* SeqPipeFinish (GPU wait) total     */
  unsigned long long encMs;  /* stitch+compressSequences total     */
};

static void HostLog(const char *fmt, ...);

static DWORD WINAPI EncWorker(void *p)
{
  EncJob *j = (EncJob *)p;
  EncShared *sh = j->sh;
  BYTE *seqs = j->seqBuf;
  for (;;) {
    WaitForSingleObject(sh->workSem, INFINITE);
    if (sh->quit) return 0;
    LONG idx = InterlockedIncrement(&sh->next) - 1;
    EncShared::Tok t = sh->toks[idx & 4095];
    unsigned pipe = t.pipeSlot;
    unsigned shm = t.shmSlot;
    const bool dbg =
        GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0;
    if (dbg) HostLog("w%lu job tok=%ld pipe=%u shm=%u\n",
        (unsigned long)GetCurrentThreadId(), (long)idx, pipe, shm);
    DWORD tA = GetTickCount();
    HRESULT hr = j->gpu->SeqPipeFinish(pipe);
    DWORD tB = GetTickCount();
    j->finMs += tB - tA;
    if (dbg) HostLog("w%lu fin tok=%ld hr=0x%lx\n",
        (unsigned long)GetCurrentThreadId(), (long)idx,
        (unsigned long)hr);
    if (FAILED(hr)) {
      SlotHdr(*j->v, shm)->err = (LONG)hr;
      SlotHdr(*j->v, shm)->outLen = 0;
      InterlockedIncrement(&sh->done);
      SetEvent(j->v->evOut[shm]);
      continue;
    }
    /* GPU pick emits per virtual segment (seg*32+lane): the same
       stitch folds lane boundaries like segment boundaries */
    unsigned nseq = StitchSegs(j->gpu->SeqPipePacked(pipe),
        j->gpu->SeqPipeCounts(pipe), j->gpu->SeqPipeCovs(pipe),
        t.inLen, j->segs, seqs);
    SetEvent(j->gpu->SeqPipeFreeEvent(pipe));
    if (dbg) HostLog("w%lu stitched tok=%ld nseq=%u\n",
        (unsigned long)GetCurrentThreadId(), (long)idx, nseq);
    BYTE *dst = SlotOut(*j->v, shm);
    BYTE *src = SlotIn(*j->v, shm);
    size_t r = ZSTD_compressSequences(j->ctx, dst, j->maxOut,
        (const ZSTD_Sequence *)(const void *)seqs, nseq, src, t.inLen);
    if (ZSTD_isError(r))
      r = ZSTD_compress(dst, j->maxOut, src, t.inLen, j->level);
    /* CPU+GPU hybrid floor: run a plain CPU zstd pass on the same
       chunk and keep whichever frame is smaller.  The workers are
       mostly idle under the GPU stream, so the extra pass is nearly
       free - it guarantees we never lose to the CPU matcher on any
       chunk (pathological parses, repcode-heavy text, etc.) */
    if (!ZSTD_isError(r) && j->ctxCpu && j->cpuScratch) {
      size_t cr = ZSTD_compress2(j->ctxCpu, j->cpuScratch,
          j->maxOut, src, t.inLen);
      if (!ZSTD_isError(cr) && cr < r) {
        memcpy(dst, j->cpuScratch, cr);
        r = cr;
      }
    }
    Z7GpuSlotHdr *h = SlotHdr(*j->v, shm);
    if (ZSTD_isError(r)) { h->err = E_FAIL; h->outLen = 0; }
    else                 { h->err = 0;      h->outLen = r; }
    MemoryBarrier();
    InterlockedIncrement(&sh->done);
    j->encMs += GetTickCount() - tB;
    SetEvent(j->v->evOut[shm]);
  }
}

static void HostLog(const char *fmt, ...)
{
  char lp[MAX_PATH];
  DWORD nl = GetEnvironmentVariableA("TEMP", lp, sizeof(lp));
  if (!nl || nl >= sizeof(lp) - 24) return;
  strcpy_s(lp + nl, sizeof(lp) - nl, "\\z7gpuhost_dbg.log");
  FILE *f = NULL;
  fopen_s(&f, lp, "a");
  if (!f) return;
  va_list ap; va_start(ap, fmt);
  vfprintf(f, fmt, ap);
  va_end(ap);
  fclose(f);
}

static void SessionCleanup(ShmView &v)
{
  /* the input area was cudaHostRegister'd for DMA - undo before the
     view is unmapped */
  if (v.slots) cudaHostUnregister(v.slots);
  if (v.ctrl) UnmapViewOfFile(v.ctrl);
  if (v.map)  CloseHandle(v.map);
  if (v.evReady) CloseHandle(v.evReady);
  if (v.evQuit)  CloseHandle(v.evQuit);
  if (v.evIn) {
    for (DWORD i = 0; i < v.nSlots; i++)
      if (v.evIn[i]) CloseHandle(v.evIn[i]);
    free(v.evIn);
  }
  if (v.evOut) {
    for (DWORD i = 0; i < v.nSlots; i++)
      if (v.evOut[i]) CloseHandle(v.evOut[i]);
    free(v.evOut);
  }
  memset(&v, 0, sizeof(v));
}

/* one compression session: attach to the parent's objects, run the
   GPU pipeline + encoder pool, stream until EOF/quit/parent-death.
   In daemon mode the CUDA context and SeqPipe stay warm for the
   next session. */
static int ServeSession(DWORD ppid, const char *sessName, bool dbg,
    bool daemon)
{
  char name[160];
  ShmView v;
  memset(&v, 0, sizeof(v));
  int rc = 0;
  CNvGpuDevice *gpu = NULL;
  DWORD made = 0;
  ZSTD_CCtx **ctxs = NULL;
  EncJob *jobs = NULL;
  HANDLE *threads = NULL;
  BYTE *seqScratch = NULL;
  EncShared sh;
  memset(&sh, 0, sizeof(sh));

  strncpy_s(name, sizeof(name), sessName, _TRUNCATE);
  char *base = name + strlen(name);

  /* ---- open the parent's objects ---- */
  if (dbg) HostLog("start name=%s\n", name);
  v.map = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
  if (!v.map) { if (dbg) HostLog("no map gle=%lu\n",
      (unsigned long)GetLastError()); rc = 3; goto done; }
  BYTE *p0 = (BYTE *)MapViewOfFile(v.map, FILE_MAP_ALL_ACCESS, 0, 0, 0);
  if (!p0) { if (dbg) HostLog("no view\n"); rc = 3; goto done; }
  v.ctrl = (Z7GpuHostCtrl *)(void *)p0;
  /* parent publishes ctrl fields before magic; wait for it */
  {
    DWORD t0 = GetTickCount();
    while (v.ctrl->magic != Z7GPU_HOST_MAGIC) {
      if (GetTickCount() - t0 > 10000) {
        if (dbg) HostLog("magic timeout\n");
        rc = 3; goto done;
      }
      Sleep(1);
    }
    if (v.ctrl->version != Z7GPU_HOST_VERSION) {
      if (dbg) HostLog("bad ver %lu\n", (unsigned long)v.ctrl->version);
      rc = 3; goto done;
    }
  }
  v.slots = p0 + 4096;   /* slot array starts on its own page */
  v.nSlots = v.ctrl->nSlots;
  if (v.nSlots == 0 || v.nSlots > 64) {
    if (dbg) HostLog("bad nslots %lu\n", (unsigned long)v.nSlots);
    rc = 3; goto done;
  }
  if (v.ctrl->chunk == 0 || v.ctrl->chunk >= ((size_t)1 << 22) ||
      v.ctrl->slotBytes < sizeof(Z7GpuSlotHdr) + v.ctrl->chunk +
          v.ctrl->maxOut) {
    if (dbg) HostLog("bad dims ch=%llu sb=%llu mo=%llu\n",
        (unsigned long long)v.ctrl->chunk,
        (unsigned long long)v.ctrl->slotBytes,
        (unsigned long long)v.ctrl->maxOut);
    rc = 3; goto done;
  }

  strcpy_s(base, sizeof(name) - (size_t)(base - name),
      Z7GPU_EVT_READY);
  v.evReady = OpenEventA(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE,
      name);
  strcpy_s(base, sizeof(name) - (size_t)(base - name),
      Z7GPU_EVT_QUIT);
  v.evQuit = OpenEventA(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE,
      name);
  if (!v.evReady || !v.evQuit) { v.ctrl->state = Z7HOST_S_FAILED;
      if (v.evReady) SetEvent(v.evReady); rc = 3; goto done; }
  v.evIn  = (HANDLE *)malloc(v.nSlots * sizeof(HANDLE));
  v.evOut = (HANDLE *)malloc(v.nSlots * sizeof(HANDLE));
  if (v.evIn) memset(v.evIn, 0, v.nSlots * sizeof(HANDLE));
  if (v.evOut) memset(v.evOut, 0, v.nSlots * sizeof(HANDLE));
  for (DWORD i = 0; v.evIn && v.evOut && i < v.nSlots; i++) {
    wsprintfA(base, "%s%u", Z7GPU_EVT_IN, i);
    v.evIn[i] = OpenEventA(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE,
        name);
    wsprintfA(base, "%s%u", Z7GPU_EVT_OUT, i);
    v.evOut[i] = OpenEventA(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE,
        name);
    if (!v.evIn[i] || !v.evOut[i]) { if (dbg) HostLog("no slot ev %lu\n",
        (unsigned long)i);
      v.ctrl->state = Z7HOST_S_FAILED;
      SetEvent(v.evReady); rc = 3; goto done; }
  }
  if (!v.evIn || !v.evOut) { v.ctrl->state = Z7HOST_S_FAILED;
      SetEvent(v.evReady); rc = 3; goto done; }
  HANDLE hParent = OpenProcess(SYNCHRONIZE, FALSE, ppid);
  if (dbg) HostLog("objects ok\n");

  v.ctrl->state = Z7HOST_S_INITING;
  SetEvent(v.evReady);

  DWORD tInit0 = GetTickCount();

  /* ---- full CUDA + pipeline init; the flaky phase ---------------- */
  gpu = CNvGpuDevice::Instance();
  gpu->Lock();
  bool ok = gpu->Init();
  DWORD tInit1 = GetTickCount();
  HRESULT hrp = S_OK;
  if (ok)
    hrp = gpu->SeqPipeInit((size_t)v.ctrl->chunk, v.ctrl->segs,
        (size_t)v.ctrl->seqStride);
  gpu->Unlock();
  DWORD tInit2 = GetTickCount();
  if (dbg)
    HostLog("init=%lums pipe=%lums ok=%d hrp=0x%lx\n",
        (unsigned long)(tInit1 - tInit0),
        (unsigned long)(tInit2 - tInit1), (int)ok,
        (unsigned long)hrp);
  if (!ok || FAILED(hrp) || gpu->SeqPipeDepth() == 0) {
    v.ctrl->state = Z7HOST_S_FAILED;
    SetEvent(v.evReady);
    rc = 3; goto done;
  }
  const unsigned K = gpu->SeqPipeDepth();

  /* encoder pool */
  const DWORD W = v.ctrl->nWorkers ? v.ctrl->nWorkers : 4;
  ctxs = (ZSTD_CCtx **)calloc(W, sizeof(ZSTD_CCtx *));
  jobs = (EncJob *)calloc(W, sizeof(EncJob));
  threads = (HANDLE *)calloc(W, sizeof(HANDLE));
  sh.next = sh.done = (LONG)v.ctrl->startTok;
  sh.toks = (EncShared::Tok *)calloc(4096, sizeof(EncShared::Tok));
  sh.workSem = CreateSemaphoreW(NULL, 0, 0x10000, NULL);
  seqScratch = (BYTE *)malloc((size_t)W * v.ctrl->seqStride);
  if (!ctxs || !jobs || !threads || !sh.toks || !sh.workSem ||
      !seqScratch) {
    v.ctrl->state = Z7HOST_S_FAILED;
    SetEvent(v.evReady);
    rc = 3; goto done;
  }
  unsigned wlog = ZSTD_WINDOWLOG_MIN;
  while (((size_t)1 << wlog) < v.ctrl->chunk &&
         wlog < ZSTD_WINDOWLOG_MAX)
    wlog++;
  made = 0;
  for (DWORD i = 0; i < W; i++) {
    ctxs[i] = ZSTD_createCCtx();
    if (!ctxs[i]) break;
    ZSTD_CCtx_setParameter(ctxs[i], ZSTD_c_compressionLevel,
        v.ctrl->level);
    ZSTD_CCtx_setParameter(ctxs[i], ZSTD_c_minMatch, ZSTD_MINMATCH_MIN);
    ZSTD_CCtx_setParameter(ctxs[i], ZSTD_c_windowLog, (int)wlog);
    ZSTD_CCtx_setParameter(ctxs[i], ZSTD_c_contentSizeFlag, 1);
    /* validation is a safety net against malformed GPU seqs; the
       kernel guarantees coverage so disabling it is a pure win -
       flip back on via env when debugging kernel changes */
    {
      char vb[8];
      DWORD vl = GetEnvironmentVariableA("Z7GPU_VALIDATE", vb, 8);
      ZSTD_CCtx_setParameter(ctxs[i], ZSTD_c_validateSequences,
          (vl && vb[0] == '1') ? 1 : 0);
    }
    ZSTD_CCtx_setParameter(ctxs[i], ZSTD_c_repcodeResolution, 1);
    EncJob *j = &jobs[i];
    j->gpu = gpu; j->pipeK = K; j->segs = v.ctrl->segs;
    j->ctx = ctxs[i]; j->seqBuf = seqScratch + (size_t)i *
        v.ctrl->seqStride;
    /* min(GPU,CPU) pass: dedicated ctx (plain defaults, only level
       + windowLog copied) + a scratch output buffer */
    j->ctxCpu = ZSTD_createCCtx();
    if (j->ctxCpu) {
      int cpulv = v.ctrl->level;
      char lb[8];
      DWORD ll = GetEnvironmentVariableA("Z7GPU_CPULV", lb, 8);
      if (ll && ll < 8) { int v9 = atoi(lb); if (v9 >= 1 && v9 <= 22)
        cpulv = v9; }
      ZSTD_CCtx_setParameter(j->ctxCpu, ZSTD_c_compressionLevel,
          cpulv);
      ZSTD_CCtx_setParameter(j->ctxCpu, ZSTD_c_windowLog, (int)wlog);
    }
    j->cpuScratch = (BYTE *)malloc((size_t)v.ctrl->maxOut);
    { char nb[8];
      DWORD nl = GetEnvironmentVariableA("Z7GPU_NOMIN", nb, 8);
      if (nl && nb[0] == '1') { j->ctxCpu = NULL; }
    }
    j->seqStride = (size_t)v.ctrl->seqStride; j->chunk =
        (size_t)v.ctrl->chunk; j->maxOut = (size_t)v.ctrl->maxOut;
    j->level = v.ctrl->level; j->v = &v; j->sh = &sh;
    threads[i] = CreateThread(NULL, 0, EncWorker, j, 0, NULL);
    if (threads[i]) made++;
  }
  if (!made) {
    v.ctrl->state = Z7HOST_S_FAILED;
    SetEvent(v.evReady);
    rc = 3; goto done;
  }

  /* registered-pinned input: the shared slot input area gets real
     DMA rate on H2D instead of staged pageable copies */
  {
    size_t inArea = (size_t)v.nSlots * v.ctrl->slotBytes;
    cudaError_t re = cudaHostRegister(v.slots, inArea,
        cudaHostRegisterDefault);
    if (re != cudaSuccess)
      (void)cudaGetLastError();
  }

  v.ctrl->state = Z7HOST_S_READY;
  SetEvent(v.evReady);
  const DWORD tReady = GetTickCount();

  /* ---- streaming loop -------------------------------------------
     chunk i goes to shm slot i%S and pipe slot pending%K; the worker
     pool drains pipe slots, encodes, signals evOut[shmSlot].        */
  volatile LONG tokHead = (LONG)v.ctrl->startTok;
  bool eof = false;
  while (!eof) {
    /* wait for the next filled shm slot (round robin) */
    DWORD s = (DWORD)(tokHead % v.nSlots);
    DWORD wr;
    if (hParent) {
      HANDLE w[3] = { v.evIn[s], v.evQuit, hParent };
      wr = WaitForMultipleObjects(3, w, FALSE, INFINITE);
    } else {
      HANDLE w[2] = { v.evIn[s], v.evQuit };
      wr = WaitForMultipleObjects(2, w, FALSE, INFINITE);
    }
    if (wr != WAIT_OBJECT_0)
      break;                                  /* quit or parent gone */
    Z7GpuSlotHdr *h = SlotHdr(v, s);
    unsigned inLen = (unsigned)h->inLen;
    if (inLen == 0) { eof = true; break; }
    /* wait until the pipe slot's previous consumer freed it */
    unsigned pipe = (unsigned)(tokHead % K);
    WaitForSingleObject(gpu->SeqPipeFreeEvent(pipe), INFINITE);
    HRESULT hr = gpu->SeqPipeSubmit(pipe, SlotIn(v, s), inLen);
    if (FAILED(hr)) { InterlockedExchange((LONG *)&v.ctrl->helperErr,
        (LONG)hr); break; }
    sh.toks[tokHead & 4095].shmSlot = s;
    sh.toks[tokHead & 4095].pipeSlot = pipe;
    sh.toks[tokHead & 4095].inLen = inLen;
    InterlockedIncrement(&tokHead);
    ReleaseSemaphore(sh.workSem, 1, NULL);
  }

  /* drain: wait until every queued token is fully processed */
  {
    DWORD t0 = GetTickCount();
    while ((volatile LONG)sh.done < tokHead &&
           GetTickCount() - t0 < 120000)
      Sleep(1);
  }
  InterlockedExchange(&sh.quit, 1);
  for (DWORD i = 0; i < made; i++) ReleaseSemaphore(sh.workSem, 1, NULL);
  for (DWORD i = 0; i < made; i++)
    WaitForSingleObject(threads[i], 5000);
  if (dbg) {
    unsigned long long fs = 0, es = 0;
    for (DWORD i = 0; i < made; i++) {
      fs += jobs[i].finMs; es += jobs[i].encMs;
    }
    double kms[3] = {0,0,0};
    gpu->SeqPipeKernelMs(kms);
    HostLog("stream=%lums gpuwait=%llums enc=%llums toks=%ld "
        "k1=%.0f k2=%.0f pk=%.0f\n",
        (unsigned long)(GetTickCount() - tReady), fs, es,
        (long)(tokHead - (LONG)v.ctrl->startTok),
        kms[0], kms[1], kms[2]);
  }

  /* eager clean teardown while the process is still healthy - but a
     daemon keeps its CUDA context and pipeline warm for the next
     session (that is the entire point of daemon mode) */
  if (!daemon && gpu) {
    gpu->Lock();
    gpu->Shutdown();
    gpu->Unlock();
  }

  v.ctrl->state = Z7HOST_S_DONE;
  SetEvent(v.evReady);
  rc = 0;

done:
  /* free per-session resources; in daemon mode this is what lets the
     next session attach cleanly instead of leaking views/handles */
  if (ctxs) {
    for (DWORD i = 0; i < made; i++)
      if (ctxs[i]) ZSTD_freeCCtx(ctxs[i]);
    free(ctxs);
  }
  if (threads) {
    for (DWORD i = 0; i < made; i++)
      if (threads[i]) CloseHandle(threads[i]);
    free(threads);
  }
  if (jobs) {
    for (DWORD i = 0; i < made; i++) {
      if (jobs[i].ctxCpu) ZSTD_freeCCtx(jobs[i].ctxCpu);
      free(jobs[i].cpuScratch);
    }
  }
  free(jobs);
  free(seqScratch);
  if (sh.toks) free(sh.toks);
  if (sh.workSem) CloseHandle(sh.workSem);
  SessionCleanup(v);
  return rc;
}

/* ---------------- persistent daemon -------------------------------
   --daemon mode: own the global rendezvous objects and serve one
   session at a time, reusing the CUDA context and SeqPipe buffers.
   Parents post their per-session control-block name through the
   channel page; on idle timeout we exit and the objects die with
   the process, so nothing stale ever survives us.  Sessions from
   dead parents end via the hParent watch inside ServeSession. */
static int DaemonLoop(bool dbg)
{
  HANDLE chan = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL,
      PAGE_READWRITE, 0, 4096, Z7GPU_DAEMON_CHAN);
  HANDLE evAlive = CreateEventA(NULL, TRUE, TRUE,
      Z7GPU_DAEMON_ALIVE);
  HANDLE evReq = CreateEventA(NULL, FALSE, FALSE,
      Z7GPU_DAEMON_REQ);
  HANDLE evAck = CreateEventA(NULL, FALSE, FALSE,
      Z7GPU_DAEMON_ACK);
  HANDLE mtx = CreateMutexA(NULL, FALSE, Z7GPU_DAEMON_MTX);
  BYTE *ch = chan ? (BYTE *)MapViewOfFile(chan,
      FILE_MAP_ALL_ACCESS, 0, 0, 4096) : NULL;
  if (!ch || !evAlive || !evReq || !evAck || !mtx) {
    if (dbg) HostLog("daemon rendezvous failed\n");
    return 3;
  }
  memset(ch, 0, 4096);
  if (dbg) HostLog("daemon up\n");
  for (;;) {
    DWORD wr = WaitForSingleObject(evReq, Z7GPU_DAEMON_IDLE_MS);
    if (wr != WAIT_OBJECT_0) break;             /* idle -> die */
    DWORD ppid = ((volatile DWORD *)ch)[0];
    char sname[160];
    memcpy(sname, ch + 8, 160);
    sname[159] = 0;
    /* publish our pid so the parent can watch for daemon death */
    ((volatile DWORD *)ch)[1] = (DWORD)GetCurrentProcessId();
    MemoryBarrier();
    SetEvent(evAck);
    ResetEvent(evAlive);                        /* now busy */
    if (dbg) HostLog("daemon session %s ppid=%lu\n", sname,
        (unsigned long)ppid);
    ServeSession(ppid, sname, dbg, true);
    SetEvent(evAlive);                          /* accepting again */
  }
  /* clean CUDA teardown while healthy, then let the named objects
     vanish with the process */
  CNvGpuDevice *gpu = CNvGpuDevice::Instance();
  gpu->Lock();
  gpu->Shutdown();
  gpu->Unlock();
  if (dbg) HostLog("daemon exit\n");
  return 0;
}

int main(int argc, char **argv)
{
  const bool dbg = GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0;
  if (argc < 2) return 2;
  if (!strcmp(argv[1], Z7GPU_DAEMON_ARG))
    return DaemonLoop(dbg);
  DWORD ppid = (DWORD)strtoul(argv[1], NULL, 10);
  char name[160];
  /* argv[2] (optional) is the full object base name; the parent
     scopes names by instance+generation so concurrent encoders and
     respawns never collide */
  if (argc > 2)
    strncpy_s(name, sizeof(name), argv[2], _TRUNCATE);
  else
    wsprintfA(name, "%s%lu.%lu", Z7GPU_HOST_PREFIX,
        (unsigned long)ppid, 0ul);
  return ServeSession(ppid, name, dbg, false);
}
