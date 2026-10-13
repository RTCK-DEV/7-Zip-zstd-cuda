// ZstdGpuDecoder.cpp - zstd decoder offloaded to the GPU via nvCOMP

#include "StdAfx.h"

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#include "ZstdGpuDecoder.h"
#include "NvGpuDevice.h"

namespace NCompress {
namespace NZSTDGPU {

/* cap for buffering the whole compressed stream for GPU decode;
   larger inputs fall back to streaming CPU decode */
static const UInt64 kGpuInputCap = (UInt64)3 << 30;
/* compressed bytes per GPU wave */
static const size_t kWaveComp = (size_t)192 << 20;
/* uncompressed staging cap per wave */
static const size_t kWaveOut = (size_t)512 << 20;

CDecoder::CDecoder():
  _ctx(NULL),
  _srcBuf(NULL),
  _dstBuf(NULL),
  _srcBufSize(ZSTD_DStreamInSize()),
  _dstBufSize(ZSTD_DStreamOutSize()),
  _processedIn(0),
  _processedOut(0),
  _readSrcPos(0),
  _readSrcLen(0),
  _comp(NULL),
  _compCap(0)
{
  _props.clear();
}

CDecoder::~CDecoder()
{
  if (_ctx) {
    ZSTD_freeDCtx(_ctx);
    MyFree(_srcBuf);
    MyFree(_dstBuf);
  }
  free(_comp);
}

HRESULT CDecoder::EnsureCtx()
{
  if (_ctx)
    return S_OK;
  _ctx = ZSTD_createDCtx();
  if (!_ctx)
    return E_OUTOFMEMORY;
  _srcBuf = MyAlloc(_srcBufSize);
  if (!_srcBuf)
    return E_OUTOFMEMORY;
  _dstBuf = MyAlloc(_dstBufSize);
  if (!_dstBuf)
    return E_OUTOFMEMORY;
  ZSTD_DCtx_setParameter(_ctx, ZSTD_d_windowLogMax, ZSTD_WINDOWLOG_MAX);
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::SetDecoderProperties2(const Byte *prop, UInt32 size))
{
  switch (size) {
  case 1:
    _props._flags = *prop;
    return S_OK;
  case 3:
  case 5:
    return S_OK;
  default:
    return E_NOTIMPL;
  }
}

HRESULT CDecoder::SetOutStreamSizeResume(const UInt64 * /*outSize*/)
{
  _processedOut = 0;
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::SetOutStreamSize(const UInt64 * /*outSize*/))
{
  _processedIn = 0;
  _readSrcPos = 0;
  _readSrcLen = 0;
  RINOK(SetOutStreamSizeResume(NULL))
  return S_OK;
}

/* decode one contiguous memory region holding complete zstd frame(s) */
HRESULT CDecoder::CpuDecodeMem(const Byte *src, size_t size,
    ISequentialOutStream *outStream, ICompressProgressInfo *progress)
{
  RINOK(EnsureCtx())
  size_t r = ZSTD_DCtx_reset(_ctx, ZSTD_reset_session_only);
  if (ZSTD_isError(r))
    return E_FAIL;

  ZSTD_inBuffer zIn;
  ZSTD_outBuffer zOut;
  zIn.src = src;
  zIn.size = size;
  zIn.pos = 0;
  size_t last = 1;

  while (zIn.pos < zIn.size) {
    zOut.dst = _dstBuf;
    zOut.size = _dstBufSize;
    zOut.pos = 0;
    last = ZSTD_decompressStream(_ctx, &zOut, &zIn);
    if (ZSTD_isError(last))
      return ERROR_INVALID_DATA;
    if (zOut.pos) {
      RINOK(WriteStream(outStream, _dstBuf, zOut.pos))
      _processedOut += zOut.pos;
      if (progress)
        RINOK(progress->SetRatioInfo(&_processedIn, &_processedOut))
    }
    if (last == 0 && zIn.pos < zIn.size) {
      r = ZSTD_DCtx_reset(_ctx, ZSTD_reset_session_only);
      if (ZSTD_isError(r))
        return E_FAIL;
    }
    if (zOut.pos == 0 && last != 0 && zIn.pos == zIn.size)
      return E_FAIL;
  }
  if (last != 0)
    return ERROR_HANDLE_EOF;
  return S_OK;
}

/* streaming CPU decode; consumes optional buffered prefix first */
HRESULT CDecoder::CpuDecodeStream(const Byte *prefix, size_t prefixLen,
    ISequentialInStream *inStream, ISequentialOutStream *outStream,
    ICompressProgressInfo *progress)
{
  RINOK(EnsureCtx())
  size_t result = ZSTD_DCtx_reset(_ctx, ZSTD_reset_session_only);
  if (ZSTD_isError(result))
    return E_FAIL;

  ZSTD_inBuffer zIn;
  ZSTD_outBuffer zOut;
  bool usingPrefix = (prefixLen != 0);

  if (usingPrefix) {
    zIn.src = prefix;
    zIn.size = prefixLen;
    zIn.pos = 0;
  } else {
    zIn.size = _srcBufSize;
    RINOK(ReadStream(inStream, _srcBuf, &zIn.size))
    _processedIn += zIn.size;
    zIn.src = _srcBuf;
    zIn.pos = 0;
  }

  for (;;) {
    for (;;) {
      zOut.dst = _dstBuf;
      zOut.size = _dstBufSize;
      zOut.pos = 0;
      result = ZSTD_decompressStream(_ctx, &zOut, &zIn);
      if (ZSTD_isError(result))
        return ERROR_INVALID_DATA;
      if (zOut.pos) {
        RINOK(WriteStream(outStream, _dstBuf, zOut.pos))
        _processedOut += zOut.pos;
        if (progress)
          RINOK(progress->SetRatioInfo(&_processedIn, &_processedOut))
      }
      if (zIn.pos == zIn.size)
        break;
      if (result == 0) {
        result = ZSTD_DCtx_reset(_ctx, ZSTD_reset_session_only);
        if (ZSTD_isError(result))
          return E_FAIL;
        if (zIn.pos < zIn.size)
          continue;
        break;
      }
    }

    if (usingPrefix) {
      /* prefix exhausted mid-frame is fine: keep going with stream */
      usingPrefix = false;
    }
    zIn.size = _srcBufSize;
    RINOK(ReadStream(inStream, _srcBuf, &zIn.size))
    _processedIn += zIn.size;
    zIn.src = _srcBuf;
    if (zIn.size == 0) {
      if (result != 0)
        return ERROR_HANDLE_EOF;
      return S_OK;
    }
    zIn.pos = 0;
  }
}

HRESULT CDecoder::GpuDecode(ISequentialInStream *inStream,
    ISequentialOutStream *outStream, const UInt64 *inSize,
    ICompressProgressInfo *progress)
{
  CNvGpuDevice *gpu = CNvGpuDevice::Instance();
  bool gpuReady = false;

  /* 1) buffer the whole compressed stream (bounded by kGpuInputCap) */
  size_t total = 0;
  if (inSize && *inSize < kGpuInputCap && *inSize > _compCap) {
    size_t want = (size_t)*inSize;
    Byte *nb = (Byte *)realloc(_comp, want);
    if (!nb)
      return E_OUTOFMEMORY;
    _comp = nb;
    _compCap = want;
  }
  bool hitCap = false;
  for (;;) {
    if (total == _compCap) {
      size_t ncap = _compCap ? _compCap * 2 : ((size_t)16 << 20);
      if (ncap > (size_t)kGpuInputCap)
        ncap = (size_t)kGpuInputCap;
      if (ncap <= _compCap) {
        /* at the cap: grow by a small margin to probe for EOF */
        Byte *nb = (Byte *)realloc(_comp, _compCap + (1 << 20));
        if (!nb)
          return E_OUTOFMEMORY;
        _comp = nb;
        size_t want = 1 << 20;
        RINOK(ReadStream(inStream, _comp + total, &want))
        if (want == 0)
          break; /* stream ended exactly at the cap */
        total += want;
        hitCap = true;
        break;
      }
      Byte *nb = (Byte *)realloc(_comp, ncap);
      if (!nb)
        return E_OUTOFMEMORY;
      _comp = nb;
      _compCap = ncap;
    }
    size_t want = _compCap - total;
    RINOK(ReadStream(inStream, _comp + total, &want))
    if (want == 0)
      break; /* eof */
    total += want;
  }
  _processedIn = total;

  if (hitCap) {
    /* stream longer than cap: CPU-decode buffered prefix + remainder */
    return CpuDecodeStream(_comp, total, inStream, outStream, progress);
  }
  if (total == 0)
    return S_OK;
  if (total < (size_t)32 << 20) {
    /* small inputs: CUDA init + transfers cost more than CPU decode */
    return CpuDecodeMem(_comp, total, outStream, progress);
  }

  /* 2) pass 1: count real frames (skip skippable ones) */
  size_t numFrames = 0;
  {
    size_t pos = 0;
    while (pos < total) {
      size_t fs = ZSTD_findFrameCompressedSize(_comp + pos, total - pos);
      if (ZSTD_isError(fs) || fs == 0)
        break;
      if (!ZSTD_isSkippableFrame(_comp + pos, fs))
        numFrames++;
      pos += fs;
    }
  }
  try { _frames.Alloc(numFrames ? numFrames * 2 : 1); }
  catch (...) { return E_OUTOFMEMORY; }

  /* pass 2: fill frame table; undecodable tail goes to the CPU */
  {
    size_t pos = 0;
    size_t fi = 0;
    while (pos < total) {
      size_t fs = ZSTD_findFrameCompressedSize(_comp + pos, total - pos);
      if (ZSTD_isError(fs) || fs == 0) {
        RINOK(CpuDecodeMem(_comp + pos, total - pos, outStream, progress))
        break;
      }
      if (!ZSTD_isSkippableFrame(_comp + pos, fs) && fi < numFrames) {
        _frames[fi * 2] = pos;
        _frames[fi * 2 + 1] = fs;
        fi++;
      }
      pos += fs;
    }
    numFrames = fi;
  }

  if (GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0) {
    fprintf(stderr, "[zstdgpu] frames=%u compBytes=%u\n",
        (unsigned)numFrames, (unsigned)total);
    fflush(stderr);
  }

  /* 3) wave loop over gpu-eligible frames */
  size_t fi = 0;
  while (fi < numFrames) {
    unsigned n = 0;
    size_t waveComp = 0;
    size_t waveOut = 0;
    size_t waveStart = fi;

    if (!gpuReady) {
      gpu->Lock();
      gpuReady = gpu->Init();
      gpu->Unlock();
      if (!gpuReady) {
        /* GPU vanished/not usable: decode every frame on the CPU */
        for (; fi < numFrames; fi++) {
          size_t off = _frames[fi * 2];
          RINOK(CpuDecodeMem(_comp + off,
              _frames[fi * 2 + 1], outStream, progress))
        }
        break;
      }
    }
    gpu->Lock();
    while (fi < numFrames && n < 1024) {
      size_t off = _frames[fi * 2];
      size_t fs  = _frames[fi * 2 + 1];
      unsigned long long cont =
          ZSTD_getFrameContentSize(_comp + off, fs);
      if (cont == ZSTD_CONTENTSIZE_UNKNOWN ||
          cont == ZSTD_CONTENTSIZE_ERROR ||
          cont > kWaveOut)
        break;
      if (n > 0 &&
          (waveComp + fs > kWaveComp || waveOut + (size_t)cont > kWaveOut))
        break;
      waveComp += fs;
      waveOut += (size_t)cont;
      n++;
      fi++;
    }
    gpu->Unlock();

    if (n == 0) {
      /* frame not gpu-eligible -> CPU decode it */
      size_t off = _frames[fi * 2];
      size_t fs  = _frames[fi * 2 + 1];
      RINOK(CpuDecodeMem(_comp + off, fs, outStream, progress))
      fi++;
      continue;
    }

    try {
      _offs.Alloc(n);
      _sizes.Alloc(n);
      _dstOffs.Alloc(n);
      _dstSizes.Alloc(n);
      _wave.Alloc(waveOut ? waveOut : 1);
    } catch (...) {
      return E_OUTOFMEMORY;
    }
    size_t firstOff = _frames[waveStart * 2];
    {
      size_t acc = 0;
      for (unsigned i = 0; i < n; i++) {
        size_t off = _frames[(waveStart + i) * 2];
        _offs[i] = off - firstOff;
        _sizes[i] = _frames[(waveStart + i) * 2 + 1];
        _dstOffs[i] = acc;
        _dstSizes[i] = (size_t)ZSTD_getFrameContentSize(_comp + off,
            _frames[(waveStart + i) * 2 + 1]);
        acc += _dstSizes[i];
      }
    }

    gpu->Lock();
    HRESULT hr = gpu->DecompressWave(_comp + firstOff, _offs, _sizes, n,
        _wave, _dstOffs, _dstSizes);
    gpu->Unlock();
    if (GetEnvironmentVariableA("Z7GPU_DEBUG", NULL, 0) != 0) {
      fprintf(stderr, "[zstdgpu] wave n=%u comp=%u out=%u hr=%x\n",
          n, (unsigned)waveComp, (unsigned)waveOut, (unsigned)hr);
      fflush(stderr);
    }

    if (hr == S_OK) {
      RINOK(WriteStream(outStream, _wave, waveOut))
      _processedOut += waveOut;
      if (progress)
        RINOK(progress->SetRatioInfo(&_processedIn, &_processedOut))
    } else {
      /* per-frame CPU fallback keeps ordering */
      for (unsigned i = 0; i < n; i++) {
        size_t off = _frames[(waveStart + i) * 2];
        RINOK(CpuDecodeMem(_comp + off,
            _frames[(waveStart + i) * 2 + 1], outStream, progress))
      }
    }
  }
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::Code(ISequentialInStream *inStream,
    ISequentialOutStream *outStream, const UInt64 *inSize,
    const UInt64 *outSize, ICompressProgressInfo *progress))
{
  RINOK(SetOutStreamSize(outSize))

  /* Default to pure-CPU decode: probing CUDA in-process is what the
     out-of-process architecture exists to avoid.  Z7GPU_GPUDEC=1
     opts back into nvCOMP batched decode. */
  if (GetEnvironmentVariableA("Z7GPU_GPUDEC", NULL, 0) != 0) {
    CNvGpuDevice *gpu = CNvGpuDevice::Instance();
    gpu->Lock();
    bool ok = gpu->Probe() && gpu->ProbeNvcomp();
    gpu->Unlock();
    if (ok)
      return GpuDecode(inStream, outStream, inSize, progress);
  }
  return CpuDecodeStream(NULL, 0, inStream, outStream, progress);
}

#ifndef Z7_NO_READ_FROM_CODER
Z7_COM7F_IMF(CDecoder::SetInStream(ISequentialInStream *inStream))
{
  _inStream = inStream;
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::ReleaseInStream())
{
  _inStream.Release();
  return S_OK;
}

Z7_COM7F_IMF(CDecoder::Read(void *data, UInt32 size, UInt32 *processedSize))
{
  if (processedSize)
    *processedSize = 0;

  RINOK(EnsureCtx())

  UInt32 totalOut = 0;
  Byte *outPtr = (Byte *)data;

  while (totalOut < size) {
    if (_readSrcPos >= _readSrcLen) {
      _readSrcLen = _srcBufSize;
      RINOK(ReadStream(_inStream, _srcBuf, &_readSrcLen))
      _processedIn += _readSrcLen;
      _readSrcPos = 0;
      if (_readSrcLen == 0)
        break;
    }

    ZSTD_inBuffer zIn;
    zIn.src = (const Byte *)_srcBuf + _readSrcPos;
    zIn.size = _readSrcLen - _readSrcPos;
    zIn.pos = 0;

    ZSTD_outBuffer zOut;
    zOut.dst = outPtr + totalOut;
    zOut.size = size - totalOut;
    zOut.pos = 0;

    size_t result = ZSTD_decompressStream(_ctx, &zOut, &zIn);
    if (ZSTD_isError(result))
      return E_FAIL;

    _readSrcPos += zIn.pos;
    totalOut += (UInt32)zOut.pos;
    _processedOut += zOut.pos;

    if (result == 0)
      ZSTD_DCtx_reset(_ctx, ZSTD_reset_session_only);
  }

  if (processedSize)
    *processedSize = totalOut;
  return S_OK;
}
#endif

Z7_COM7F_IMF(CDecoder::SetNumberOfThreads(UInt32 /*numThreads*/))
{
  return S_OK;
}

}}
