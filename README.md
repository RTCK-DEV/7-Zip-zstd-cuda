# 7-Zip Zstandard — CUDA-accelerated edition

[English](#english) | [日本語](#日本語) | [中文](#中文)

---

## English

An experimental 7-Zip codec fork that accelerates Zstandard compression
with a custom LZ77 parser running on an NVIDIA GPU (CUDA). Drop
`zstdgpu.dll` into `Codecs/` and the GPU takes over match finding and
sequence construction through the usual interface:

```
7z a -m0=ZSTDGPU archive.7z files\     # GPU compression
7z x archive.7z                     # ordinary extraction (standard zstd frames)
```

Output is a concatenation of standard Zstandard frames, so archives
**decompress in any stock zstd/7-Zip environment**. Without a GPU or the
helper process it falls back to the CPU encoder automatically.

### Architecture

- **`SeqGen.cu`** — candidate kernel (multi-stage hash probes + direct
  small-offset probes + forward propagation) and a per-segment
  warp-cooperative pick kernel (rep-aware best-of-5 scoring, lazy
  lookahead, backward extension). An experimental forward-DP parser is
  selectable via `Z7GPU_OPT=1`.
- **`NvGpuDevice`** — chunked pipeline (streams/slots), reverse-adaptive
  512 KiB/2 MiB window selection from measured first-chunk
  compressibility, chunk-proportional hash-table scaling.
- **`z7gpuhost.exe`** — a persistent out-of-process helper owning the
  CUDA context. Per chunk it compares the GPU-sequence output against
  CPU zstd output and keeps the smaller frame (`Z7GPU_NOMIN=1` disables).
- **Encoding** — GPU sequences are framed on the CPU with
  `ZSTD_compressSequences`; FSE/Huffman entropy coding is upstream zstd.

Details: [Codec_zstdgpu/README.md](CPP/7zip/Bundles/Codec_zstdgpu/README.md)

### Benchmarks (RTX 3070, CUDA 12.9)

| Input | CPU zstd -3 | zstdgpu | Wall |
|---|---:|---:|---:|
| text, 159 MiB | 2.03 MiB | **1.77 MiB (-13%)** | ~0.6 s |
| corpus, 482 MiB | 60.3 MiB | **47.9 MiB (-21%)** | ~1.2 s |
| low-entropy | — | parity (per-chunk min) | — |

Round-trips verified on text/corpus/binary/random/empty/1-byte inputs
and under concurrent compression.

### Build

Requires the CUDA toolkit (12.9) and VS2022:

```
cd CPP/7zip/Bundles/Codec_zstdgpu
nmake            # x64\zstdgpu.dll and x64\z7gpuhost.exe
```

Deploy `zstdgpu.dll` to `Codecs/`; `z7gpuhost.exe` to `Codecs/` or next
to the DLL.

### Key environment variables

| Variable | Value | Meaning |
|---|---|---|
| `Z7GPU_ENGINE` | `cpu` | force CPU fallback, no GPU |
| `Z7GPU_CHUNKKB` | 64–4095 | fixed chunk size (unset = adaptive 512K/2M) |
| `Z7GPU_SEGS` | 32–1024 | parse parallelism (default 32) |
| `Z7GPU_NOMIN` | `1` | disable per-chunk CPU comparison |
| `Z7GPU_OPT` | `1` | experimental DP parser (research) |
| `Z7GPU_DEBUG` | `1` | diagnostics (`%TEMP%\z7gpuhost_dbg.log`) |

### License

GNU LGPL v2.1-or-later, same as 7-Zip ZS.

---

## 日本語

7-Zip 向けの Zstandard コーデックを、NVIDIA GPU (CUDA) 上で走る独自
LZ77 パーサーで高速化した実験的フォークです。
`zstdgpu.dll` を `Codecs/` に置くだけで、いつもの操作のまま GPU が
マッチ探索とシーケンス構築を引き受けます:

```
7z a -m0=ZSTDGPU archive.7z files\     # GPU 圧縮
7z x archive.7z                     # 普通に解凍 (標準 zstd フレーム)
```

出力は標準 Zstandard フレームの連結なので、**普通の zstd/7-Zip
環境でそのまま解凍できます**。GPU や helper プロセスが無ければ
CPU エンコーダへ自動フォールバックします。

### アーキテクチャ

- **`SeqGen.cu`** — 候補生成カーネル (多段ハッシュプローブ +
  小オフセット直接プローブ + 前方伝播) と、セグメント単位の
  warp 協調 pick カーネル (rep-aware best-of-5 スコアリング、
  lazy 先読み、後方延長)。実験的な前向き DP パーサーも
  `Z7GPU_OPT=1` で選択可能。
- **`NvGpuDevice`** — チャンク化パイプライン (ストリーム/スロット)、
  先頭チャンクの実測圧縮率から 512KiB/2MiB ウィンドウを自動選択
  する逆適応、チャンク比例のハッシュテーブル拡大。
- **`z7gpuhost.exe`** — GPU コンテキストを持つ別プロセス常駐
  helper。GPU シーケンス出力と CPU zstd 出力をチャンク毎に比較し、
  小さいフレームを採用 (`Z7GPU_NOMIN=1` で無効化)。
- **エンコード** — GPU 生成シーケンスを CPU 側
  `ZSTD_compressSequences` でフレーム化。FSE/Huffman のエントロピー
  符号は zstd 本家のもの。

詳細は [Codec_zstdgpu/README.md](CPP/7zip/Bundles/Codec_zstdgpu/README.md)。

### ベンチマーク (RTX 3070, CUDA 12.9)

| 入力 | CPU zstd -3 | zstdgpu | Wall |
|---|---:|---:|---:|
| text, 159 MiB | 2.03 MiB | **1.77 MiB (-13%)** | ~0.6 s |
| corpus, 482 MiB | 60.3 MiB | **47.9 MiB (-21%)** | ~1.2 s |
| 低圧縮性データ | — | 互角 (per-chunk min が効く) | — |

往復一致は text/corpus/binary/random/空・1 バイト等のエッジ
ケース、並行圧縮で検証済み。

### ビルド

CUDA toolkit (12.9) と VS2022 が必要です:

```
cd CPP/7zip/Bundles/Codec_zstdgpu
nmake            # x64\zstdgpu.dll と x64\z7gpuhost.exe
```

`zstdgpu.dll` → `Codecs/`、`z7gpuhost.exe` → `Codecs/` または
DLL と同じ場所に配置。

### 主な環境変数

| 変数 | 値 | 意味 |
|---|---|---|
| `Z7GPU_ENGINE` | `cpu` | GPU を使わず CPU フォールバック強制 |
| `Z7GPU_CHUNKKB` | 64–4095 | チャンク固定 (未設定 = 逆適応 512K/2M) |
| `Z7GPU_SEGS` | 32–1024 | パース並列度 (既定 32) |
| `Z7GPU_NOMIN` | `1` | per-chunk CPU 比較の無効化 |
| `Z7GPU_OPT` | `1` | 実験 DP パーサー (研究用) |
| `Z7GPU_DEBUG` | `1` | 診断ログ (`%TEMP%\z7gpuhost_dbg.log`) |

### ライセンス

ベースの 7-Zip ZS と同じ GNU LGPL v2.1-or-later。

---

## 中文

这是一个实验性的 7-Zip 编解码器分支：用运行在 NVIDIA GPU (CUDA) 上的
自研 LZ77 解析器加速 Zstandard 压缩。只需把 `zstdgpu.dll` 放入
`Codecs/` 目录，GPU 就会在你习惯的操作方式下接管匹配查找与序列
构建：

```
7z a -m0=ZSTDGPU archive.7z files\     # GPU 压缩
7z x archive.7z                     # 正常解压 (标准 zstd 帧)
```

输出是标准 Zstandard 帧的串接，因此**在任何原版 zstd/7-Zip 环境中
都能正常解压**。没有 GPU 或 helper 进程时会自动回退到 CPU 编码器。

### 架构

- **`SeqGen.cu`** — 候选生成 kernel（多段哈希探测 + 小偏移直接探测 +
  前向传播）以及按段划分的 warp 协作 pick kernel（rep 感知的
  best-of-5 打分、lazy 前瞻、反向延伸）。实验性的前向 DP 解析器可
  通过 `Z7GPU_OPT=1` 启用。
- **`NvGpuDevice`** — 分块流水线（stream/slot）、根据首个块的实测
  压缩率在 512KiB/2MiB 窗口间自动切换的逆向自适应机制、随块大小
  伸缩的哈希表。
- **`z7gpuhost.exe`** — 持有 GPU 上下文的常驻独立 helper 进程。每个
  块比较 GPU 序列输出与 CPU zstd 输出，取较小的帧
  （`Z7GPU_NOMIN=1` 关闭）。
- **编码** — GPU 生成的序列在 CPU 上经 `ZSTD_compressSequences` 成帧；
  FSE/Huffman 熵编码沿用上游 zstd。

详见 [Codec_zstdgpu/README.md](CPP/7zip/Bundles/Codec_zstdgpu/README.md)。

### 基准测试 (RTX 3070, CUDA 12.9)

| 输入 | CPU zstd -3 | zstdgpu | 耗时 |
|---|---:|---:|---:|
| text, 159 MiB | 2.03 MiB | **1.77 MiB (-13%)** | ~0.6 s |
| corpus, 482 MiB | 60.3 MiB | **47.9 MiB (-21%)** | ~1.2 s |
| 低可压缩数据 | — | 持平 (per-chunk min) | — |

已在 text/corpus/二进制/随机/空文件/1 字节等边界情况及并发压缩下
完成往返一致性验证。

### 构建

需要 CUDA toolkit (12.9) 与 VS2022：

```
cd CPP/7zip/Bundles/Codec_zstdgpu
nmake            # 产出 x64\zstdgpu.dll 与 x64\z7gpuhost.exe
```

`zstdgpu.dll` 放到 `Codecs/`；`z7gpuhost.exe` 放到 `Codecs/` 或与
DLL 同目录。

### 主要环境变量

| 变量 | 取值 | 含义 |
|---|---|---|
| `Z7GPU_ENGINE` | `cpu` | 强制 CPU 回退，不用 GPU |
| `Z7GPU_CHUNKKB` | 64–4095 | 固定块大小 (未设置 = 自适应 512K/2M) |
| `Z7GPU_SEGS` | 32–1024 | 解析并行度 (默认 32) |
| `Z7GPU_NOMIN` | `1` | 关闭每块 CPU 对比 |
| `Z7GPU_OPT` | `1` | 实验性 DP 解析器 (研究用) |
| `Z7GPU_DEBUG` | `1` | 诊断日志 (`%TEMP%\z7gpuhost_dbg.log`) |

### 许可证

与上游 7-Zip ZS 相同：GNU LGPL v2.1-or-later。
