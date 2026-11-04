# 第2回 LoRA 比較コード

このコードは、講座の公式Notebookに含まれるFAQ、データ分割、生成・採点関数をSHA-256で照合して使用します。学習器と結果検証器はこのディレクトリにあります。

## 実行環境と注意

提出に使ったのは Apple M5 / MPS、Python 3.13.12、torch 2.11.0 の環境です。受講者本人がGoogle Colab/T4で学習したとは主張していません。保存済み結果は、このMac上での代理実行と独立検証を記録しています。

`requirements-mps.txt` はmacOSのCPU/MPS用です。Google Colab/T4にはインストールしないでください。T4で実行する場合は、公式NotebookのセットアップセルにあるCUDA対応パッケージを使ってください。CUDA版PyTorchへの切り替えやQLoRAは、この提出では検証していません。

モデルは公開リポジトリから `token=False` で取得します。Hugging Faceの保存済み認証情報を暗黙送信しません。

## 再現手順（macOS MPS）

リポジトリのルートで実行します。出力先は空の新規ディレクトリにしてください。仮想環境と大きな出力は `/tmp` に置き、重み・中間checkpoint・モデルキャッシュを提出リポジトリへ紛れ込ませないようにします。

```sh
cd /path/to/lecture-ai-engineering-2026
AIE_DAY2_DIR="$(mktemp -d /tmp/aie2026-day2.XXXXXX)"
python3.13 -m venv "$AIE_DAY2_DIR/venv"
PYTHON_BIN="$AIE_DAY2_DIR/venv/bin/python"
"$PYTHON_BIN" -m pip install -r lectures/02/code/requirements-mps.txt
"$PYTHON_BIN" -c 'import torch, transformers, peft, trl, accelerate, datasets; print(torch.__version__, torch.backends.mps.is_available())'
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 "$PYTHON_BIN" lectures/02/code/run_lora.py --source-notebook notebooks/lora_format_tuning.ipynb --output "$AIE_DAY2_DIR/baseline_epochs20" --device mps
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 "$PYTHON_BIN" lectures/02/code/run_lora.py --source-notebook notebooks/lora_format_tuning.ipynb --output "$AIE_DAY2_DIR/changed_epochs10" --device mps --epochs 10
"$PYTHON_BIN" lectures/02/code/validate_results.py --source-notebook notebooks/lora_format_tuning.ipynb "$AIE_DAY2_DIR/baseline_epochs20" "$AIE_DAY2_DIR/changed_epochs10"
```

比較する設定の差は `num_train_epochs` のみです。各runは別のプロセスで基盤モデルから開始します。実行コマンドは保存済み成果物の再生成用であり、Colab/T4での同一結果を保証しません。

## 提出済み結果

- [比較表と制約](../results/comparison.json)
- [20epoch run](../results/baseline_epochs20/) と [10epoch run](../results/changed_epochs10/)（質問ごとの出力、データ、設定、loss/時間、検査対象ファイルのmanifestを含む）
- [独立再採点・設定・ハッシュ検証](../results/validation.json)

両runを `verified: true` としています。公開結果にはモデル重みと中間checkpointを含めていませんが、掲載済みの出力・データ・設定・loss記録については、このディレクトリのvalidatorを使って再採点と整合性検査を実行できます。未知の質問への汎化や安全性を保証するものではありません。
