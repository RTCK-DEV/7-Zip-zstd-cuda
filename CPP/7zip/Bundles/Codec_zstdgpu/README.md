# Codec_zstdgpu — CUDA-accelerated Zstandard codec

An experimental 7-Zip codec that runs the LZ77 stage of Zstandard
compression on an NVIDIA GPU. Match candidates are generated in a CUDA
kernel (`SeqGen.cu`, multi-probe hash buckets + direct small-offset probes
+ forward propagation), parsed into sequences by a warp-cooperative
greedy/lazy picker with repcode-aware scoring, and emitted as **ordinary
Zstandard frames** via `ZSTD_compressSequences`. Output is a normal `.7z`
with method `ZSTDGPU` — it shares stock zstd's method id `0x4F71101`,
so archives still decompress with the ordinary ZSTD codec.

The GPU work lives in a separate daemon process (`z7gpuhost.exe`, spawned
on first use and reused) so the codec DLL stays dependency-free. If no GPU
or helper is found, compression falls back to the plain CPU zstd encoder.

## Build

Requires the CUDA toolkit (developed against CUDA 12.9) and VS2022:

```
cd CPP/7zip/Bundles/Codec_zstdgpu
nmake            # builds x64\zstdgpu.dll and x64\z7gpuhost.exe
```

Deploy `zstdgpu.dll` into `Codecs/` next to `7z.exe`/`7zFM.exe`, and
`z7gpuhost.exe` either into `Codecs/` or beside the codec DLL.

## Usage

```
7z a -m0=ZSTDGPU archive.7z files\     # GPU-accelerated compression
7z x archive.7z                     # ordinary extraction
```

## Environment variables

| Variable | Value | Meaning |
|---|---|---|
| `Z7GPU_ENGINE` | `seq` (default) / `cpu` | GPU pipeline / forced CPU fallback |
| `Z7GPU_GPUDEC` | `1` | opt-in GPU decompression (default: CPU decode) |
| `Z7GPU_CHUNKKB` | 64–4095 | fixed chunk size; unset = adaptive window (start at 2 MiB, retune to 512 KiB if the first chunk is not compressible) |
| `Z7GPU_SEGS` | 32–1024 | parse segments per chunk (default 32) |
| `Z7GPU_SLOTS` | 4–64 | shared-memory ring slots (default 16) |
| `Z7GPU_WORKERS` | 1–32 | helper zstd encoder threads (default: physical cores) |
| `Z7GPU_NOMIN` | `1` | disable per-chunk min(GPU seqs, CPU zstd) frame selection |
| `Z7GPU_CPULV` | 1–22 | level for the per-chunk CPU comparison |
| `Z7GPU_VALIDATE` | `1` | re-enable `ZSTD_c_validateSequences` (debug) |
| `Z7GPU_OPT` | `1` | experimental forward-DP parser (research; slower and worse ratio than the default picker) |
| `Z7GPU_LITC`, `Z7GPU_SEQB`, `Z7GPU_REPC` | int | DP cost knobs in 1/256-bit units |
| `Z7GPU_DEBUG` | `1` | diagnostic logging to stderr / `%TEMP%\z7gpuhost_dbg.log` |

## Results (RTX 3070, CUDA 12.9)

| Input | CPU zstd -3 | zstdgpu | Wall |
|---|---:|---:|---:|
| text, 159 MiB | 2.03 MiB | **1.77 MiB (-13%)** | ~0.6 s |
| corpus, 482 MiB | 60.3 MiB | **47.9 MiB (-21%)** | ~1.2 s |
| near-incompressible | — | parity | — |

Round-trip verified on text/corpus/binary/random/edge-case inputs and
under concurrent compression.
