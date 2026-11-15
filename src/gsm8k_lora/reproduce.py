"""Recompute selection, numeric grading, and summaries from saved records on CPU."""

import argparse
import csv
from decimal import Decimal
from pathlib import Path
import shutil
import tempfile

from .common import DEFAULT_CONFIG, extract_answer, fingerprint, read_json, read_jsonl, write_json
from .select_data import select
from .summarize import summarize


def reproduce(artifact_root, report_path=None):
    root = Path(artifact_root).resolve()
    metadata = read_json(root / "data" / "metadata.json")
    candidates = read_jsonl(root / "data" / "candidates.jsonl")
    validation = read_jsonl(root / "data" / "validation.jsonl")
    test = read_jsonl(root / "data" / "test.jsonl")
    actual_fingerprint = fingerprint({"candidates": candidates, "validation": validation, "test": test,
                                      "model_revision": metadata["model_revision"],
                                      "dataset_revision": metadata["dataset_revision"]})
    if actual_fingerprint != metadata["fingerprint"]:
        raise ValueError("Prepared data fingerprint mismatch")
    config = dict(metadata["config"])
    names = ["base"] + [f"{method}_{seed}" for seed in config["seeds"] for method in ("random", "moderate")]
    verified = []
    with tempfile.TemporaryDirectory(prefix="gsm8k-reproduce-") as temporary:
        temporary = Path(temporary)
        data, runs = temporary / "data", temporary / "runs"
        data.mkdir()
        for name in ("metadata.json", "candidates.jsonl"):
            shutil.copy2(root / "data" / name, data / name)
        shutil.copytree(root / "data" / "scores", data / "scores")
        config.update(data_dir=str(data), output_dir=str(runs))
        select(config)
        for seed in config["seeds"]:
            for method in ("random", "moderate"):
                name = f"{method}_{seed}"
                expected = read_json(root / "data" / "selections" / f"{name}.json")
                actual = read_json(data / "selections" / f"{name}.json")
                if actual != expected:
                    raise ValueError(f"Selection differs: {name}")
                training = read_json(root / "runs" / name / "train.json")
                if training["debug"] or training["actual"] != {
                    "T": expected["summary"]["T"], "T_y": expected["summary"]["T_y"],
                    "exposure_examples": expected["summary"]["count"],
                    "update_steps": expected["expected_update_steps"],
                }:
                    raise ValueError(f"Training budget mismatch: {name}")
        for name in names:
            for split, source_rows in (("validation", validation), ("test", test)):
                evaluation = root / "runs" / name / f"eval_{split}"
                records = read_jsonl(evaluation / "predictions.jsonl")
                expected_ids = {row["id"] for row in source_rows}
                if len(records) != len(expected_ids) or {row["id"] for row in records} != expected_ids:
                    raise ValueError(f"Incomplete prediction IDs: {name}/{split}")
                references = {row["id"]: row for row in source_rows}
                for record in records:
                    reference = references[record["id"]]
                    if any(record[key] != reference[key] for key in ("question", "answer")):
                        raise ValueError(f"Reference mismatch: {name}/{record['id']}")
                    gold = extract_answer(record["answer"], strict=True)
                    prediction = extract_answer(record["output"])
                    strict = extract_answer(record["output"], strict=True)
                    if gold is None or Decimal(record["gold"]) != gold:
                        raise ValueError(f"Gold mismatch: {name}/{record['id']}")
                    values = {"prediction": str(prediction) if prediction is not None else None,
                              "strict_prediction": str(strict) if strict is not None else None,
                              "correct": prediction == gold, "strict_correct": strict == gold}
                    if any(record[key] != value for key, value in values.items()):
                        raise ValueError(f"Numeric grading differs: {name}/{record['id']}")
                metrics = read_json(evaluation / "metrics.json")
                computed = {
                    "count": len(records),
                    "accuracy": sum(row["correct"] for row in records) / len(records),
                    "strict_accuracy": sum(row["strict_correct"] for row in records) / len(records),
                    "strict_format_rate": sum(row["strict_prediction"] is not None for row in records) / len(records),
                    "parse_failure_rate": sum(row["prediction"] is None for row in records) / len(records),
                    "hit_limit_rate": sum(row["hit_limit"] for row in records) / len(records),
                    "mean_generated_tokens": sum(row["generated_tokens"] for row in records) / len(records),
                }
                if any(metrics[key] != value for key, value in computed.items()):
                    raise ValueError(f"Metric mismatch: {name}/{split}")
                shutil.copytree(evaluation, runs / name / f"eval_{split}")
                verified.append({"method": name, "split": split, **computed})
            if name != "base":
                shutil.copy2(root / "runs" / name / "train.json", runs / name / "train.json")
        for split in ("validation", "test"):
            summarize(config, split)
            if read_json(runs / f"comparison_{split}.json") != read_json(root / "runs" / f"comparison_{split}.json"):
                raise ValueError(f"Paired bootstrap comparison differs: {split}")
            with (runs / f"summary_{split}.csv").open(encoding="utf-8", newline="") as handle:
                actual = list(csv.DictReader(handle))
            with (root / "runs" / f"summary_{split}.csv").open(encoding="utf-8", newline="") as handle:
                expected = list(csv.DictReader(handle))
            if actual != expected:
                raise ValueError(f"Summary differs: {split}")
    report = {"status": "passed", "data_fingerprint": actual_fingerprint,
              "candidate_count": len(candidates), "verified_predictions": sum(row["count"] for row in verified),
              "checks": ["data fingerprint", "deterministic selection", "actual training budgets",
                         "prediction IDs and references", "numeric grading", "metrics", "paired bootstrap", "CSV summaries"],
              "evaluations": verified}
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, report)
    print(f"Verified {report['verified_predictions']} saved predictions and both seed-13 selections.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_CONFIG.parent.parent / "artifacts")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reproduce(args.artifacts, args.report)


if __name__ == "__main__":
    main()
