// ZstdGpuEncoder.h - zstd encoder with GPU match finding
//
// Registers the stock ZSTD method id (0x4F71101). The custom CUDA
// kernel (SeqGen.cu) generates LZ77 sequences per chunk on the GPU;
// the host feeds them to ZSTD_compressSequences() so each chunk becomes
// a standard zstd frame readable by any zstd-capable 7-Zip.
// Z7GPU_ENGINE=nvcomp selects the nvCOMP compressor instead;
// Z7GPU_ENGINE=cpu forces the CPU path. Falls back to CPU zstd when
// no CUDA device is present.

#ifndef __ZSTD_GPU_ENCODER_H
#define __ZSTD_GPU_ENCODER_H

#define ZSTD_STATIC_LINKING_ONLY
#include "../../../C/Alloc.h"
#include "../../../C/Threads.h"
#include "../../../C/zstd/zstd.h"

#include "../../Common/Common.h"
#include "../../Common/MyCom.h"
#include "../../Common/MyBuffer.h"
#include "../ICoder.h"
#include "../Common/StreamUtils.h"

#ifndef Z7_EXTRACT_ONLY
namespace NCompress {
namespace NZSTDGPU {

struct CProps
{
  CProps() { clear (); }
  void clear ()
  {
    memset(this, 0, sizeof (*this));
    _ver_major = ZSTD_VERSION_MAJOR;
    _ver_minor = ZSTD_VERSION_MINOR;
    _level = 3;
  }

  Byte _ver_major;
  Byte _ver_minor;
  Byte _level;
  Byte _reserved[2];
};

Z7_CLASS_IMP_COM_5(
  CEncoder,
  ICompressCoder,
  ICompressSetCoderMt,
  ICompressSetCoderProperties,
  ICompressSetCoderPropertiesOpt,
  ICompressWriteCoderProperties
)
public:
  CProps _props;

  ZSTD_CCtx *_ctx;
  void  *_srcBuf;
  void  *_dstBuf;
  size_t _srcBufSize;
  size_t _dstBufSize;

  UInt64 _processedIn;
  UInt64 _processedOut;
  UInt32 _numThreads;
  Int32  _Level;
  UInt64 unpackSize;

  /* gpu wave state */
  CBuffer<Byte> _inBuf;    /* staging: waveN * chunkSize          */
  CBuffer<Byte> _outBuf;   /* staging: waveN * maxOut             */
  CBuffer<size_t> _sizes;  /* chunk sizes, capacity waveN          */
  CBuffer<Byte> _seqBuf;   /* staging: waveN * seqStride          */
  CBuffer<unsigned> _seqCounts; /* per-chunk sequence counts       */
  CBuffer<size_t> _outSizes; /* per-chunk compressed sizes         */
  ZSTD_CCtx *_seqCtx;      /* cctx for ZSTD_compressSequences      */
  ZSTD_CCtx **_seqCtxs;    /* per-worker cctx pool                 */
  unsigned _numSeqCtxs;
  size_t _chunkSize;
  unsigned _waveN;
  void *_pinnedBuf;        /* _inBuf pinned for real-DMA H2D      */

  size_t ChunkSize() const;
  HRESULT CodeCpu(ISequentialInStream *inStream, ISequentialOutStream *outStream,
      ICompressProgressInfo *progress);
  HRESULT CodeGpu(ISequentialInStream *inStream, ISequentialOutStream *outStream,
      ICompressProgressInfo *progress);
  HRESULT CodeGpuSeq(ISequentialInStream *inStream, ISequentialOutStream *outStream,
      ICompressProgressInfo *progress);
  /* out-of-process variant: CUDA lives in z7gpuhost.exe, we drive it
     through shared memory. No CUDA calls in this process. */
  HRESULT CodeGpuHostSeq(ISequentialInStream *inStream, ISequentialOutStream *outStream,
      ICompressProgressInfo *progress);

  CEncoder();
  ~CEncoder();
};

}}
#endif
#endif
