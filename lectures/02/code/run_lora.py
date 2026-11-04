#!/usr/bin/env python3
"""公式FAQ/評価を変更せず、量子化なしLoRAをMPS/CUDA/CPUで比較する。"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time

# 公開モデルだけを匿名取得する。保存済み認証情報を暗黙送信しない。
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

SOURCE_SHA256 = "2edcfb6d1461dabdbfea8242a7ae4ec7c231f4d2e5cec61920a4317ce8694b00"
UPSTREAM_COMMIT = "662cddd52724b330f3f4cd15123d1e55cb474874"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_body(notebook, index):
    return ast.parse("".join(notebook["cells"][index]["source"]))


def assigned_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def source_definitions(notebook, index, namespace, functions=(), assignments=()):
    """完全性をSHA-256確認した公式セルから、指定した定義だけを採用する。"""
    body = [n for n in source_body(notebook, index).body
            if (isinstance(n, ast.FunctionDef) and n.name in functions)
            or assigned_name(n) in assignments]
    found = {n.name if isinstance(n, ast.FunctionDef) else assigned_name(n) for n in body}
    if found != set(functions) | set(assignments):
        raise ValueError(f"source definitions changed: {index}: {found}")
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 f"official_notebook_cell_{index}", "exec"), namespace)


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"出力先は空の新規ディレクトリにしてください: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if sha(args.source_notebook) != SOURCE_SHA256:
        raise SystemExit("原本NotebookのSHA-256が既定の公式版と異なります")
    notebook = json.loads(args.source_notebook.read_text(encoding="utf-8"))
    namespace = {"Path": Path, "os": os, "json": json, "time": time, "re": re, "torch": torch}
    config_node = next(n for n in source_body(notebook, 5).body if assigned_name(n) == "CONFIG")
    config = ast.literal_eval(config_node.value)
    config.pop("drive_dir")
    config["work_dir"] = str(args.output)
    # 本番では比較したいこの1項目以外は公式設定を変更しない。
    if args.epochs is not None:
        config["num_train_epochs"] = args.epochs
    namespace.update(CONFIG=config, WORK=args.output)
    exec(compile("".join(notebook["cells"][7]["source"]), "official_notebook_cell_7", "exec"), namespace)
    source_definitions(notebook, 9, namespace, functions=("build_messages", "generate"), assignments=("SYSTEM",))
    source_definitions(notebook, 10, namespace, functions=("check_format", "check_content", "mark", "evaluate"))
    source_definitions(notebook, 12, namespace, functions=("to_chat",))
    source_definitions(notebook, 17, namespace, functions=("classify",), assignments=("KNOWLEDGE_QUESTIONS",))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPSを使用できません")
    set_seed(config["seed"])
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "numpy")}
    write_json(args.output / "runtime.json", {"python": platform.python_version(), "device": device,
               "versions": versions, "dtype": "float32", "source_commit": UPSTREAM_COMMIT,
               "source_notebook_sha256": SOURCE_SHA256, "runner_sha256": sha(__file__),
               "smoke_test": args.smoke, "quantization": "none"})
    print(json.dumps({"phase": "load_model", "device": device, "versions": versions}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"],
                                               token=False, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["revision"],
            dtype=torch.float32, token=False, trust_remote_code=False).to(device)
    model.config.use_cache = True
    model.eval()
    namespace["tok"] = tokenizer
    base_memory = model.get_memory_footprint()
    if args.smoke:
        namespace["test"] = namespace["test"][:1]
        namespace["KNOWLEDGE_QUESTIONS"] = []
    evaluate = namespace["evaluate"]
    before_summary, before_rows = evaluate(model, "before")
    print(json.dumps({"phase": "before_complete", **before_summary}), flush=True)
    train_ds = Dataset.from_list([namespace["to_chat"](e) for e in namespace["train"]])
    valid_ds = Dataset.from_list([namespace["to_chat"](e) for e in namespace["valid"]])
    model.config.use_cache = False
    lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    sft = SFTConfig(output_dir=str(args.output / "ckpt"), num_train_epochs=config["num_train_epochs"],
            max_steps=2 if args.smoke else -1, per_device_train_batch_size=4,
            gradient_accumulation_steps=2, learning_rate=config["lr"], lr_scheduler_type="cosine",
            warmup_steps=config["warmup_steps"], logging_steps=1 if args.smoke else 10,
            save_steps=10, save_total_limit=2, fp16=False, bf16=False,
            max_length=config["max_seq_len"], packing=False, report_to="none", seed=config["seed"])
    trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=valid_ds,
                         peft_config=lora, processing_class=tokenizer)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    write_json(args.output / "sft_config.json", trainer.args.to_dict())
    print(json.dumps({"phase": "train", "trainable_parameters": trainable,
                      "all_parameters": total, "steps_expected": 2 if args.smoke else
                      math.ceil(len(train_ds) / 8) * config["num_train_epochs"]}), flush=True)
    sync(device)
    start = time.perf_counter()
    train_output = trainer.train()
    sync(device)
    train_seconds = time.perf_counter() - start
    trainer.model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    trainer.model.config.use_cache = True
    trainer.model.eval()
    write_json(args.output / "train_metrics.json", {"train_seconds": train_seconds,
              "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
              "training_loss": train_output.training_loss, "log_history": trainer.state.log_history,
              "mps_allocated_bytes": torch.mps.current_allocated_memory() if device == "mps" else None,
              "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if device == "mps" else None,
              "base_model_bytes": base_memory, "trainable_parameters": trainable, "all_parameters": total})
    after_summary, after_rows = evaluate(trainer.model, "after")
    regressions = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
    knowledge_rows = []
    # goldはtest専用21問の採点表であり、学習用の根拠あり質問を含まない。
    # 追加観察の表示ラベルのみを、公式GROUNDEDの質問一覧に照合する。
    known_faq_questions = {q for templates, _ in namespace["GROUNDED"].values() for q in templates}
    known_faq_questions.update(q for templates in namespace["TEST_EXTRA_GROUNDED"].values() for q in templates)
    for question in namespace["KNOWLEDGE_QUESTIONS"]:
        example = namespace["make_example"](question, "", True)
        with trainer.model.disable_adapter():
            before_raw = namespace["generate"](trainer.model, example)
        after_raw = namespace["generate"](trainer.model, example)
        knowledge_rows.append({"質問": question, "FAQ にある": question in known_faq_questions,
            "学習前の分類": namespace["classify"](before_raw), "学習前の出力": before_raw,
            "学習後の分類": namespace["classify"](after_raw), "学習後の出力": after_raw})
    result = {"config": {k: v for k, v in config.items() if k != "work_dir"}, "versions": versions,
              "method": "LoRA（量子化なし）", "before": before_summary, "after": after_summary,
              "regressed": regressions, "knowledge": knowledge_rows,
              "executed_training_myself": not args.smoke, "training_executed": True,
              "smoke_test": args.smoke, "train_seconds": train_seconds,
              "global_step": trainer.state.global_step, "final_training_loss": next(
                  (e["loss"] for e in reversed(trainer.state.log_history) if "loss" in e), None),
              "training_loss_average": train_output.training_loss,
              "runtime_device": device, "source_commit": UPSTREAM_COMMIT,
              "source_notebook_sha256": SOURCE_SHA256}
    write_json(args.output / "result.json", result)
    files = [p for p in args.output.rglob("*") if p.is_file() and "ckpt" not in p.relative_to(args.output).parts]
    write_json(args.output / "manifest.json", {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files})
    (args.output / "completed.flag").write_text("smoke_complete\n" if args.smoke else "training_and_evaluation_complete\n", encoding="utf-8")
    print(json.dumps({"phase": "complete", "smoke": args.smoke, "train_seconds": train_seconds,
                      "before": before_summary, "after": after_summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
