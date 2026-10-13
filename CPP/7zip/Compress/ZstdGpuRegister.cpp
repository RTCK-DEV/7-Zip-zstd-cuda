// ZstdGpuRegister.cpp - register the GPU-accelerated zstd codec
//
// Same method id as stock ZSTD (0x4F71101): output is a sequence of
// standard zstd frames, so archives stay compatible with 7-Zip ZS.

#include "StdAfx.h"

#include "../Common/RegisterCodec.h"

#include "ZstdGpuDecoder.h"

#ifndef Z7_EXTRACT_ONLY
#include "ZstdGpuEncoder.h"
#endif

REGISTER_CODEC_E(
  ZSTD,
  NCompress::NZSTDGPU::CDecoder(),
  NCompress::NZSTDGPU::CEncoder(),
  0x4F71101, "ZSTD")
