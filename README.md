# 7-Zip Zstandard — CUDA-accelerated edition

7-Zip 向けの Zstandard コーデックを、NVIDIA GPU (CUDA) 上で走る独自
LZ77 パーサーで高速化した実験的フォークです。
`zstdgpu.dll` を `Codecs/` に置くだけで、いつもの操作のまま GPU が
マッチ探索とシーケンス構築を引き受けます:

```
7z a -m0=ZSTD archive.7z files\     # GPU 圧縮
7z x archive.7z                     # 普通に解凍 (標準 zstd フレーム)
```

出力は標準 Zstandard フレームの連結なので、**普通の zstd/7-Zip
環境でそのまま解凍できます**。GPU や helper プロセスが無ければ
CPU エンコーダへ自動フォールバックします。

## アーキテクチャ

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

## ベンチマーク (RTX 3070, CUDA 12.9)

| 入力 | CPU zstd -3 | zstdgpu | Wall |
|---|---:|---:|---:|
| text, 159 MiB | 2.03 MiB | **1.77 MiB (-13%)** | ~0.6 s |
| corpus, 482 MiB | 60.3 MiB | **47.9 MiB (-21%)** | ~1.2 s |
| 低圧縮性データ | — | 互角 (per-chunk min が効く) | — |

往復一致は text/corpus/binary/random/空・1 バイト等のエッジ
ケース、並行圧縮で検証済み。

## ビルド

CUDA toolkit (12.9) と VS2022 が必要です:

```
cd CPP/7zip/Bundles/Codec_zstdgpu
nmake            # x64\zstdgpu.dll と x64\z7gpuhost.exe
```

`zstdgpu.dll` → `Codecs/`、`z7gpuhost.exe` → `Codecs/` または
DLL と同じ場所に配置。

## 主な環境変数

| 変数 | 値 | 意味 |
|---|---|---|
| `Z7GPU_ENGINE` | `cpu` | GPU を使わず CPU フォールバック強制 |
| `Z7GPU_CHUNKKB` | 64–4095 | チャンク固定 (未設定 = 逆適応 512K/2M) |
| `Z7GPU_SEGS` | 32–1024 | パース並列度 (既定 32) |
| `Z7GPU_NOMIN` | `1` | per-chunk CPU 比較の無効化 |
| `Z7GPU_OPT` | `1` | 実験 DP パーサー (研究用) |
| `Z7GPU_DEBUG` | `1` | 診断ログ (`%TEMP%\z7gpuhost_dbg.log`) |

## ライセンス

ベースの 7-Zip ZS と同じ GNU LGPL v2.1-or-later。
