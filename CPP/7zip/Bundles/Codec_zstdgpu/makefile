PROG = zstdgpu.dll
DEF_FILE = ../../Compress/Codec.def

NVCOMP_DIR = E:\dev\nvcomp-wheel\libnvcomp\nvidia\libnvcomp
CUDA_DIR = $(CUDA_PATH)

CFLAGS = $(CFLAGS) -DZSTD_LEGACY_SUPPORT -DZSTD_MULTITHREAD -wd4819 -I"$(NVCOMP_DIR)\include" -I"$(CUDA_DIR)\include"
LFLAGS = $(LFLAGS) /DELAYLOAD:nvcomp64_5.dll /LIBPATH:"$(NVCOMP_DIR)\lib" /LIBPATH:"$(CUDA_DIR)\lib\x64"
LIBS = $(LIBS) delayimp.lib nvcomp.lib cudart_static.lib

7ZIP_COMMON_OBJS = \
  $O\StreamUtils.obj \

WIN_OBJS = \
  $O\System.obj \

COMPRESS_OBJS = \
  $O\CodecExports.obj \
  $O\DllExportsCompress.obj \

C_OBJS = \
  $O\Alloc.obj \
  $O\Threads.obj \

COMPRESS_OBJS = $(COMPRESS_OBJS) \
  $O\NvGpuDevice.obj \
  $O\ZstdGpuDecoder.obj \
  $O\ZstdGpuEncoder.obj \
  $O\ZstdGpuRegister.obj \

ZSTD_OBJS = \
  $O\debug.obj \
  $O\entropy_common.obj \
  $O\error_private.obj \
  $O\fse_compress.obj \
  $O\fse_decompress.obj \
  $O\hist.obj \
  $O\huf_compress.obj \
  $O\huf_decompress.obj \
  $O\pool.obj \
  $O\threading.obj \
  $O\xxhash.obj \
  $O\zstd_common.obj \
  $O\zstd_compress.obj \
  $O\zstd_compress_literals.obj \
  $O\zstd_compress_sequences.obj \
  $O\zstd_compress_superblock.obj \
  $O\zstd_ddict.obj \
  $O\zstd_decompress_block.obj \
  $O\zstd_decompress.obj \
  $O\zstd_double_fast.obj \
  $O\zstd_fast.obj \
  $O\zstd_lazy.obj \
  $O\zstd_ldm.obj \
  $O\zstdmt_compress.obj \
  $O\zstd_opt.obj \
  $O\zstd_preSplit.obj \
  $O\zstd_v01.obj \
  $O\zstd_v02.obj \
  $O\zstd_v03.obj \
  $O\zstd_v04.obj \
  $O\zstd_v05.obj \
  $O\zstd_v06.obj \
  $O\zstd_v07.obj \

ZSTDMT_OBJS = \
  $O\zstd-mt_threading.obj \

# custom CUDA object: nmake expands << inline files at parse time, so
# LIBS must carry SeqGen.obj before Build.mak's link rule is read.
LIBS = $(LIBS) $O\SeqGen.obj

!include "../../7zip.mak"

# --- custom CUDA sequence generator -----------------------------------
# compiled with nvcc; linked via LIBS (OBJS lists have fixed per-dir rules)
NVCC = "$(CUDA_DIR)\bin\nvcc.exe"
NVCCFLAGS = -c -O3 -std=c++11 \
  -gencode=arch=compute_75,code=sm_75 \
  -gencode=arch=compute_80,code=sm_80 \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_89,code=sm_89 \
  -gencode=arch=compute_90,code=sm_90 \
  -gencode=arch=compute_90,code=compute_90

$(PROGPATH): $O\SeqGen.obj

$O\SeqGen.obj: ../../Compress/SeqGen.cu
	$(NVCC) $(NVCCFLAGS) -o $@ ..\..\Compress\SeqGen.cu

# --- out-of-process CUDA host ------------------------------------------
# z7gpuhost.exe owns the CUDA context + sequence pipeline; the codec DLL
# only talks shared memory to it.  Built without the 7zip PCH/CRT glue so
# it can run standalone; reuses the already-built zstd objects in $O.

HOST_CFLAGS = -nologo -c -W3 -EHsc -Gy -MT -MP -O2 -GR- -Gw \
  -DZSTD_LEGACY_SUPPORT -DZSTD_MULTITHREAD \
  -I"$(NVCOMP_DIR)\include" -I"$(CUDA_DIR)\include" -I..\..\..\..\C\zstd

HOST_OBJS = \
  $O\z7gpuhost.obj \
  $O\NvGpuDeviceHost.obj \
  $O\SeqGen.obj \
  $(ZSTD_OBJS) \
  $(ZSTDMT_OBJS) \
  $O\Alloc.obj \
  $O\Threads.obj \

$O\z7gpuhost.exe: $(HOST_OBJS)
	link -nologo -OPT:REF -OPT:ICF -INCREMENTAL:NO /LARGEADDRESSAWARE \
	  /SUBSYSTEM:CONSOLE,6.02 -out:$@ $(HOST_OBJS) \
	  delayimp.lib nvcomp.lib cudart_static.lib \
	  /DELAYLOAD:nvcomp64_5.dll \
	  /LIBPATH:"$(NVCOMP_DIR)\lib" /LIBPATH:"$(CUDA_DIR)\lib\x64" \
	  kernel32.lib user32.lib advapi32.lib

$O\z7gpuhost.obj: z7gpuhost.cpp ../../Compress/GpuHostShared.h ../../Compress/NvGpuDevice.h
	$(CC) $(HOST_CFLAGS) -Fo$O\z7gpuhost.obj z7gpuhost.cpp

$O\NvGpuDeviceHost.obj: ../../Compress/NvGpuDevice.cpp ../../Compress/NvGpuDevice.h
	$(CC) $(HOST_CFLAGS) -Fo$O\NvGpuDeviceHost.obj ..\..\Compress\NvGpuDevice.cpp
