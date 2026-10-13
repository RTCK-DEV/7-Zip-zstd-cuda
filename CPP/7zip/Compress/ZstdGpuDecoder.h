// ZstdGpuDecoder.h - zstd decoder offloaded to the GPU via nvCOMP
//
// Handles the stock ZSTD method id (0x4F71101). Concatenated zstd
// frames are split and decompressed batched on the GPU; frames the GPU
// rejects (and any non-GPU host) are decoded by the embedded CPU zstd.

#ifndef __ZSTD_GPU_DECODER_H
#define __ZSTD_GPU_DECODER_H

#define ZSTD_STATIC_LINKING_ONLY
#include "../../../C/Alloc.h"
#include "../../../C/zstd/zstd.h"
#include "../../../C/zstd/zstd_errors.h"

#include "../../Windows/System.h"
#include "../../Common/Common.h"
#include "../../Common/MyCom.h"
#include "../../Common/MyBuffer.h"
#include "../ICoder.h"
#include "../Common/StreamUtils.h"
#include "../Common/RegisterCodec.h"
#include "../Common/ProgressMt.h"

namespace NCompress {
namespace NZSTDGPU {

struct DProps
{
  DProps() { clear (); }
  void clear () { _flags = 0; }
  Byte _flags;
};

class CDecoder Z7_final:
  public ICompressCoder,
  public ICompressSetDecoderProperties2,
  public ICompressSetCoderMt,
  public ICompressSetOutStreamSize,
#ifndef Z7_NO_READ_FROM_CODER
  public ICompressSetInStream,
  public ISequentialInStream,
#endif
  public CMyUnknownImp
{
  CMyComPtr < ISequentialInStream > _inStream;

public:
  DProps _props;

  ZSTD_DCtx *_ctx;
  void  *_srcBuf;
  void  *_dstBuf;
  size_t _srcBufSize;
  size_t _dstBufSize;

  UInt64 _processedIn;
  UInt64 _processedOut;
  size_t _readSrcPos;
  size_t _readSrcLen;

  Byte   *_comp;           /* whole compressed stream (realloc'd)  */
  size_t  _compCap;
  CBuffer<size_t> _frames; /* pairs: offset, size                  */
  CBuffer<size_t> _offs;   /* wave work arrays                     */
  CBuffer<size_t> _sizes;
  CBuffer<size_t> _dstOffs;
  CBuffer<size_t> _dstSizes;
  CBuffer<Byte>   _wave;   /* decompressed wave staging            */

  HRESULT EnsureCtx();
  HRESULT CpuDecodeMem(const Byte *src, size_t size,
      ISequentialOutStream *outStream, ICompressProgressInfo *progress);
  HRESULT CpuDecodeStream(const Byte *prefix, size_t prefixLen,
      ISequentialInStream *inStream, ISequentialOutStream *outStream,
      ICompressProgressInfo *progress);
  HRESULT GpuDecode(ISequentialInStream *inStream,
      ISequentialOutStream *outStream, const UInt64 *inSize,
      ICompressProgressInfo *progress);
  HRESULT SetOutStreamSizeResume(const UInt64 *outSize);

  Z7_COM_QI_BEGIN2(ICompressCoder)
  Z7_COM_QI_ENTRY(ICompressSetDecoderProperties2)
  Z7_COM_QI_ENTRY(ICompressSetCoderMt)
  Z7_COM_QI_ENTRY(ICompressSetOutStreamSize)
#ifndef Z7_NO_READ_FROM_CODER
  Z7_COM_QI_ENTRY(ICompressSetInStream)
  Z7_COM_QI_ENTRY(ISequentialInStream)
#endif
  Z7_COM_QI_END
  Z7_COM_ADDREF_RELEASE

  Z7_IFACE_COM7_IMP(ICompressCoder)
  Z7_IFACE_COM7_IMP(ICompressSetDecoderProperties2)
public:
  Z7_IFACE_COM7_IMP(ICompressSetCoderMt)

  Z7_IFACE_COM7_IMP(ICompressSetOutStreamSize)
#ifndef Z7_NO_READ_FROM_CODER
  Z7_IFACE_COM7_IMP(ICompressSetInStream)
  Z7_IFACE_COM7_IMP(ISequentialInStream)
  UInt64 GetInputProcessedSize() const { return _processedIn; }
#endif

public:
  CDecoder();
  ~CDecoder();
};

}}
#endif
