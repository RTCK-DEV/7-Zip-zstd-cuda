/* GpuHostShared.h - shared-memory protocol between the 7z codec
   (in the 7z process) and z7gpuhost.exe (out-of-process CUDA worker).

   The CUDA driver init path intermittently fast-fails inside
   nvcuda64 on this driver branch (HEAP_FAILURE_INVALID_ENTRY, raised
   as 0xC0000374 inside cuInit's nested module loads).  There is no
   in-process mitigation: the fault is a fast-fail, uncatchable.
   Running all CUDA work in a helper process isolates the failure -
   if the helper dies during init the codec falls back to the CPU
   path before any input is consumed, and the archive stays correct.
*/
#pragma once

#include <windows.h>

#define Z7GPU_HOST_MAGIC    0x5A374750u   /* 'Z7GP' */
#define Z7GPU_HOST_VERSION  1u

/* object name prefix; real names append the parent pid */
#define Z7GPU_HOST_PREFIX   "Local\\Z7GpuHost."

/* control block at offset 0 of the mapping */
struct Z7GpuHostCtrl {
  volatile DWORD magic;        /* Z7GPU_HOST_MAGIC by parent       */
  volatile DWORD state;        /* Z7HOST_S_* below                 */
  DWORD version;
  DWORD nSlots;                /* ring depth                       */
  unsigned __int64 chunk;      /* raw bytes per chunk              */
  DWORD segs;                  /* segments per chunk               */
  DWORD pad0;
  unsigned __int64 seqStride;  /* stitched sequence capacity bytes */
  unsigned __int64 maxOut;     /* ZSTD_compressBound(chunk)        */
  __int32 level;               /* zstd compression level           */
  DWORD nWorkers;              /* zstd encoder threads in helper   */
  unsigned __int64 totalIn;    /* stats: fed bytes                 */
  unsigned __int64 totalOut;   /* stats: produced bytes            */
  volatile LONG done;          /* helper: 1 = stream finished      */
  volatile LONG helperErr;     /* helper: last HRESULT-ish error   */
  unsigned __int64 slotBytes;  /* stride between slots             */
  unsigned __int64 startTok;   /* first token index (0 normally;    */
                               /* nextOut when respawning mid-run)  */
};

/* ctrl.state values */
enum {
  Z7HOST_S_BOOT = 0,      /* helper starting                       */
  Z7HOST_S_INITING = 1,   /* helper running cuInit/pipeline setup  */
  Z7HOST_S_READY = 2,     /* helper fully initialized, streaming   */
  Z7HOST_S_FAILED = 3,    /* helper init failed (codec falls back) */
  Z7HOST_S_DONE = 4       /* helper saw EOF marker and finished    */
};

/* per-slot region, parent-owned fields first then helper-owned */
struct Z7GpuSlotHdr {
  volatile DWORD seq;         /* request sequence number            */
  volatile DWORD phase;       /* 0 empty, 1 filling, 2 ready,
                                 3 compressing, 4 drained           */
  unsigned __int64 inLen;     /* 0 = EOF marker                     */
  unsigned __int64 outLen;    /* compressed frame bytes             */
  volatile LONG err;          /* nonzero on helper-side failure     */
  DWORD pad;
  /* payload follows: inLen bytes of input, then a maxOut output
     area; layout is  in[0..chunk) | out[0..maxOut)                 */
};

/* event suffixes appended to Z7GPU_HOST_PREFIX + pid */
#define Z7GPU_EVT_READY   ".ready"   /* helper->parent: state change */
#define Z7GPU_EVT_IN      ".in."     /* +slot index                  */
#define Z7GPU_EVT_OUT     ".out."    /* +slot index                  */
#define Z7GPU_EVT_QUIT    ".quit"    /* parent->helper: terminate    */

#define Z7GPU_READY_TIMEOUT_MS 30000

/* ---- persistent-daemon rendezvous -------------------------------
   A long-lived helper (--daemon) owns these objects; they die with
   the process so a stale daemon never leaves them behind.  A parent
   posts its per-session control-block name through the channel
   instead of spawning a fresh process: the daemon reuses its CUDA
   context and pipeline, hiding all ~110ms of init after the first
   run.  Layout of the Chan page: [DWORD ppid][DWORD daemonPid]
   [char ctrlName[160]]. */
#define Z7GPU_DAEMON_CHAN   "Local\\Z7GpuDaemon.Chan"
#define Z7GPU_DAEMON_ALIVE  "Local\\Z7GpuDaemon.Alive"
#define Z7GPU_DAEMON_REQ    "Local\\Z7GpuDaemon.Req"
#define Z7GPU_DAEMON_ACK    "Local\\Z7GpuDaemon.Ack"
#define Z7GPU_DAEMON_MTX    "Local\\Z7GpuDaemon.Mtx"
#define Z7GPU_DAEMON_ARG    "--daemon"
/* idle exit: daemon shuts itself down when no session arrives */
#define Z7GPU_DAEMON_IDLE_MS (10 * 60 * 1000)

/* helper exe name (must sit next to the codec dll) */
#define Z7GPU_HOST_EXE  "z7gpuhost.exe"
