#!/usr/bin/env python3
"""公開する学習結果を、保存済み原文・正解から独立に再採点する。"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

from run_lora import SOURCE_SHA256, sha, source_definitions, source_body, assigned_name


def official_dataset(notebook):
    """SHA確認済みセル7のデータ定義を再構築する（表示・保存部分は実行しない）。"""
    body = [node for node in source_body(notebook, 7).body
            if not isinstance(node, ast.Expr) and assigned_name(node) != "_"]
    namespace = {"json": json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 "official_dataset_validation", "exec"), namespace)
    return namespace


def official_config(notebook):
    node = next(node for node in source_body(notebook, 5).body if assigned_name(node) == "CONFIG")
    return {key: value for key, value in ast.literal_eval(node.value).items()
            if key not in ("work_dir", "drive_dir")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_run(root, notebook, smoke=False):
    errors = []

    def require(ok, message):
        if not ok:
            errors.append(message)

    ns = {"json": json}
    source_definitions(notebook, 10, ns, functions=("check_format", "check_content", "mark", "evaluate"))
    result = read(root / "result.json")
    gold = read(root / "gold.json")
    test = [json.loads(row) for row in (root / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [json.loads(row) for row in (root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    original = official_dataset(notebook)
    for name, saved in (("train", train), ("test", test), ("gold", gold)):
        require(saved == original[name], f"{name} differs from official source")
    questions = [r["question"] for r in train + test]
    require(len(questions) == len(set(questions)), "train/test question overlap")
    require(len(train) == 14 and len(test) == 21 and len(gold) == 21, "dataset size mismatch")
    require(sum(r["needs_review"] for r in train) == 4, "train needs_review histogram mismatch")
    require(sum(r["needs_review"] for r in test) == 6, "test needs_review histogram mismatch")
    eval_count = 1 if smoke else 21
    for phase in ("before", "after"):
        saved = read(root / f"eval_{phase}.json")
        require(len(saved["rows"]) == eval_count, f"{phase} row count")
        require([r["question"] for r in saved["rows"]] == [r["question"] for r in test[:eval_count]], f"{phase} ordering")
        totals = {"format_rate": 0, "content_rate": 0, "both_rate": 0}
        for row in saved["rows"]:
            require(isinstance(row["raw"], str), f"{phase}: raw output missing")
            ok_format, reason, obj = ns["check_format"](row["raw"])
            loose = obj
            if not ok_format:
                match = re.search(r"\{.*\}", row["raw"], re.S)
                try:
                    loose = json.loads(match.group(0)) if match else None
                except Exception:
                    loose = None
            ok_content = ns["check_content"](loose if isinstance(loose, dict) else None, gold[row["question"]])
            require(row["format_ok"] == ok_format and row["format_reason"] == reason,
                    f"{phase}: format score mismatch")
            require(row["content_ok"] == ok_content, f"{phase}: content score mismatch")
            require(row["both_ok"] == (ok_format and ok_content), f"{phase}: combined score mismatch")
            for key, value in (("format_rate", ok_format), ("content_rate", ok_content), ("both_rate", ok_format and ok_content)):
                totals[key] += int(value)
        for key, total in totals.items():
            require(abs(saved["summary"][key] - total / eval_count) < 1e-12, f"{phase}: {key} summary mismatch")
            require(saved["summary"][key] == result[phase][key], f"{phase}: result {key} mismatch")
    training = read(root / "train_metrics.json")
    expected_steps = 2 if smoke else 2 * result["config"]["num_train_epochs"]
    require(training["global_step"] == expected_steps == result["global_step"], "step count mismatch")
    require(math.isfinite(training["training_loss"]), "nonfinite training loss")
    require(math.isfinite(result["final_training_loss"]), "nonfinite final logged loss")
    require(result["train_seconds"] > 0, "invalid elapsed training time")
    require(result["source_notebook_sha256"] == SOURCE_SHA256, "source mismatch")
    require(result["executed_training_myself"] == (not smoke), "smoke/full execution flag mismatch")
    require((root / "completed.flag").is_file(), "completion flag missing")
    for name, expected in read(root / "manifest.json").items():
        path = root / name
        require(path.is_file(), f"missing manifest file: {name}")
        if path.is_file():
            require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], f"hash mismatch: {name}")
    adapter = read(root / "adapter/adapter_config.json")
    require(adapter["r"] == result["config"]["lora_r"], "adapter rank mismatch")
    require(adapter["base_model_name_or_path"] == result["config"]["base_model"], "adapter base model mismatch")
    if not smoke:
        control = next((row for row in result["knowledge"] if row["質問"] == "課題の提出期限はいつですか"), None)
        require(control is not None and control["FAQ にある"] is True, "grounded knowledge control mislabeled")
    return {"verified": not errors, "errors": errors, "smoke_test": smoke,
            "global_step": training["global_step"], "train_seconds": result["train_seconds"],
            "before": result["before"], "after": result["after"], "result_sha256": sha(root / "result.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    if sha(args.source_notebook) != SOURCE_SHA256:
        raise SystemExit("original notebook hash mismatch")
    notebook = read(args.source_notebook)
    reports = [check_run(root, notebook, args.smoke) for root in args.runs]
    if len(args.runs) == 2 and not args.smoke:
        configs = [read(root / "result.json")["config"] for root in args.runs]
        differences = [key for key in configs[0] if configs[0][key] != configs[1][key]]
        sfts = [read(root / "sft_config.json") for root in args.runs]
        sft_differences = [key for key in sfts[0] if sfts[0][key] != sfts[1][key]]
        runtimes = [read(root / "runtime.json") for root in args.runs]
        data_hashes_equal = all(sha(args.runs[0] / name) == sha(args.runs[1] / name)
                               for name in ("train.jsonl", "test.jsonl", "gold.json"))
        baseline_original = any(config == official_config(notebook) for config in configs)
        allowed_sft_differences = {"output_dir", "num_train_epochs"}
        valid_comparison = (len(differences) == 1 and differences[0] == "num_train_epochs"
                            and baseline_original and data_hashes_equal and runtimes[0] == runtimes[1]
                            and set(sft_differences) <= allowed_sft_differences)
        reports.append({"config_differences": differences, "sft_differences": sft_differences,
                        "baseline_matches_official": baseline_original,
                        "dataset_hashes_equal": data_hashes_equal,
                        "runtime_and_runner_hash_equal": runtimes[0] == runtimes[1],
                        "verified": valid_comparison})
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(r["verified"] for r in reports) else 1)


if __name__ == "__main__":
    main()
