# 第2回 LoRA 比較コード

このコードは、講座の公式Notebookに含まれるFAQ、データ分割、生成・採点関数をSHA-256で照合して使用します。学習器と結果検証器はこのディレクトリにあります。

## 実行環境と注意

提出に使ったのは Apple M5 / MPS、Python 3.13.12、torch 2.11.0 の環境です。受講者本人がGoogle Colab/T4で学習したとは主張していません。保存済み結果は、このMac上での代理実行と独立検証を記録しています。

`requirements-mps.txt` はmacOSのCPU/MPS用です。Google Colab/T4にはインストールしないでください。T4で実行する場合は、公式NotebookのセットアップセルにあるCUDA対応パッケージを使ってください。CUDA版PyTorchへの切り替えやQLoRAは、この提出では検証していません。

モデルは公開リポジトリから `token=False` で取得します。Hugging Faceの保存済み認証情報を暗黙送信しません。

## 再現手順（macOS MPS）

リポジトリのルートで実行します。出力先は空の新規ディレクトリにしてください。重み・中間checkpoint・モデルキャッシュは提出リポジトリに含めません。

```sh
python3.13 -m venv .venv-lora
. .venv-lora/bin/activate
python -m pip install -r lectures/02/code/requirements-mps.txt
python -c 'import torch, transformers, peft, trl, accelerate, datasets; print(torch.__version__, torch.backends.mps.is_available())'
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python lectures/02/code/run_lora.py --source-notebook notebooks/lora_format_tuning.ipynb --output .study_ops/cache/aie2026_day2/baseline_epochs20 --device mps
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python lectures/02/code/run_lora.py --source-notebook notebooks/lora_format_tuning.ipynb --output .study_ops/cache/aie2026_day2/changed_epochs10 --device mps --epochs 10
python lectures/02/code/validate_results.py --source-notebook notebooks/lora_format_tuning.ipynb .study_ops/cache/aie2026_day2/baseline_epochs20 .study_ops/cache/aie2026_day2/changed_epochs10
```

比較する設定の差は `num_train_epochs` のみです。各runは別のプロセスで基盤モデルから開始します。実行コマンドは保存済み成果物の再生成用であり、Colab/T4での同一結果を保証しません。

## 提出済み結果

- [比較表と制約](../results/comparison.json)
- [独立再採点・設定・ハッシュ検証](../results/validation.json)

どちらも両runを `verified: true` としています。これらは保存済み結果と手順の整合性を検査するもので、未知の質問への汎化や安全性を保証するものではありません。
