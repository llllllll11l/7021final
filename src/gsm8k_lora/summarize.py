import argparse
import csv
import random
from pathlib import Path

from .common import DEFAULT_CONFIG, load_config, load_prepared, read_json, read_jsonl, write_json, write_jsonl


def summarize(config, split):
    root, metadata = load_prepared(config)
    output = Path(config["output_dir"])
    names = ["base"] + [f"{method}_{seed}" for seed in config["seeds"] for method in ("random", "moderate")]
    scoring = read_json(root / "scores" / "metadata.json")["cost"]
    rows, predictions = [], {}
    comparison_config = None
    for name in names:
        evaluation = read_json(output / name / f"eval_{split}" / "metrics.json")
        if evaluation["data_fingerprint"] != metadata["fingerprint"]:
            raise ValueError(f"{name} 评估数据不匹配。")
        current_config = {"train": evaluation["config"]["train"], "generation": evaluation["config"]["generation"]}
        if comparison_config is not None and current_config != comparison_config:
            raise ValueError("各组训练/生成配置不一致，不能合并为公平比较。")
        comparison_config = current_config
        base = name == "base"
        training = None if base else read_json(output / name / "train.json")
        records = read_jsonl(output / name / f"eval_{split}" / "predictions.jsonl")
        predictions[name] = {row["id"]: row for row in records}
        expected = metadata["official_test_count"] if split == "test" else metadata["validation_count"]
        if len(predictions[name]) != expected or len(records) != expected or evaluation["count"] != expected:
            raise ValueError(f"{name} 预测分母不完整。")
        scoring_cost = scoring if name.startswith("moderate") else {"seconds": 0, "peak_allocated_gib": 0}
        train_cost = training["cost"] if training else {"seconds": 0, "peak_allocated_gib": 0}
        row = {"method": name, "seed": evaluation["seed"],
               "T": training["actual"]["T"] if training else 0,
               "T_y": training["actual"]["T_y"] if training else 0,
               "unique_samples": training["unique_samples"] if training else 0,
               "update_steps": training["actual"]["update_steps"] if training else 0,
               **{key: evaluation[key] for key in ("accuracy", "strict_accuracy", "strict_format_rate",
                                                  "parse_failure_rate", "hit_limit_rate", "mean_generated_tokens")},
               "score_minutes": scoring_cost["seconds"] / 60, "train_minutes": train_cost["seconds"] / 60,
               "eval_minutes": evaluation["cost"]["seconds"] / 60,
               "peak_allocated_gib": max(scoring_cost["peak_allocated_gib"], train_cost["peak_allocated_gib"],
                                         evaluation["cost"]["peak_allocated_gib"])}
        rows.append(row)
    with (output / f"summary_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons, examples = [], []
    for seed in config["seeds"]:
        baseline, moderate = predictions[f"random_{seed}"], predictions[f"moderate_{seed}"]
        if set(baseline) != set(moderate) or set(baseline) != set(predictions["base"]):
            raise ValueError("配对比较的题目 ID 不一致。")
        ids = sorted(baseline)
        differences = [int(moderate[identifier]["correct"]) - int(baseline[identifier]["correct"]) for identifier in ids]
        rng = random.Random(2026)
        bootstrap = sorted(sum(rng.choices(differences, k=len(ids))) / len(ids) * 100 for _ in range(2000))
        comparisons.append({"seed": seed, "C_minus_B_pp": sum(differences) / len(ids) * 100,
                            "paired_bootstrap_95_ci_pp": [bootstrap[49], bootstrap[1949]]})
        categories = {"all_wrong": [], "B_right_C_wrong": [], "C_right_B_wrong": []}
        for identifier in ids:
            base_correct = predictions["base"][identifier]["correct"]
            random_correct, moderate_correct = baseline[identifier]["correct"], moderate[identifier]["correct"]
            category = ("B_right_C_wrong" if random_correct and not moderate_correct else
                        "C_right_B_wrong" if moderate_correct and not random_correct else
                        "all_wrong" if not (base_correct or random_correct or moderate_correct) else None)
            if category:
                categories[category].append(identifier)
        for category, candidates in categories.items():
            rng.shuffle(candidates)
            for identifier in candidates[:4 if len(config["seeds"]) > 1 else 8]:
                examples.append({"id": identifier, "seed": seed, "category": category,
                                 "question": baseline[identifier]["question"], "gold": baseline[identifier]["gold"],
                                 "A": predictions["base"][identifier]["output"],
                                 "B": baseline[identifier]["output"], "C": moderate[identifier]["output"],
                                 "human_error_type": "", "human_notes": ""})
    write_jsonl(output / f"error_cases_{split}.jsonl", examples)
    write_json(output / f"comparison_{split}.json", {
        "pairs": comparisons,
        "mean_accuracy": {method: sum(row["accuracy"] for row in rows if row["method"].startswith(method)) / len(config["seeds"])
                          for method in ("random", "moderate")},
        "mean_C_minus_B_pp": sum(row["C_minus_B_pp"] for row in comparisons) / len(comparisons),
        "recorded_formal_compute_minutes": scoring["seconds"] / 60 + sum(row["train_minutes"] + row["eval_minutes"] for row in rows),
        "cost_note": "表内每个 C 均含一次全池评分；实际正式运行只计一次缓存评分。不含加载、验证、试跑、排障及人时。",
        "bootstrap_note": "按题目配对重采样 2000 次，仅反映测试题不确定性，不替代训练种子重复。",
        "case_selection": "每种子按固定 RNG 从三种错误类别各取最多 4 题；单种子各取最多 8 题。",
    })
    for row in rows:
        print(f"{row['method']:15s} accuracy={row['accuracy']:.4f} T={row['T']} train={row['train_minutes']:.2f} min")
    print(comparisons)


def main():
    parser = argparse.ArgumentParser(description="汇总 A/B/C、配对 bootstrap 和人工错误分析样例")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    summarize(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
