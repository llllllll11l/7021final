"""Run fresh scoring, seed-13 LoRA training, and complete held-out evaluations."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reproduction-rtx5080"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--train-only", action="store_true", help="Score and train; omit held-out generation")
    parser.add_argument("--resume", action="store_true", help="Skip stages already recorded as successful")
    parser.add_argument("--batch-size", type=int, choices=(2, 4), help="Use 2 for lower training memory, preserving effective batch 32")
    args = parser.parse_args()
    root, source = args.output.resolve(), args.prepared.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=args.resume)
    logs = root / "logs"
    logs.mkdir(exist_ok=args.resume)
    config_path = root / "config.json"
    if args.resume:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.batch_size and config["train"]["batch_size"] != args.batch_size:
            raise ValueError("Choose a new output directory when changing the training batch size")
    else:
        shutil.copytree(source, root / "data", ignore=shutil.ignore_patterns("scores", "selections", ".ipynb_checkpoints"))
        config = json.loads((source / "metadata.json").read_text(encoding="utf-8"))["config"]
        config.update(data_dir="data", output_dir="runs")
        if args.batch_size:
            config["train"]["batch_size"] = args.batch_size
            config["train"]["gradient_accumulation_steps"] = 32 // args.batch_size
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_DISABLE_XET": "1", "PYTHONUTF8": "1",
           "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    stages = [("score", "gsm8k_lora.run", ["score"]), ("select", "gsm8k_lora.select_data", [])]
    for method in ("random", "moderate"):
        stages.append((f"train_{method}", "gsm8k_lora.run", ["train", "--method", method, "--seed", "13"]))
    if not args.train_only:
        for split in ("validation", "test"):
            for method in ("base", "random", "moderate"):
                stages.append((f"{split}_{method}", "gsm8k_lora.run",
                               ["eval", "--method", method, "--seed", "13", "--split", split]))
            stages.append((f"summary_{split}", "gsm8k_lora.summarize", ["--split", split]))
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if args.resume and progress_path.exists() else []
    completed = {row["stage"] for row in progress if row["exit_code"] == 0}
    for name, module, options in stages:
        if name in completed:
            print(f"SKIP {name}: completed", flush=True)
            continue
        started = time.time()
        print(f"START {name}", flush=True)
        command = [sys.executable, "-u", "-m", module, "--config", str(config_path), *options]
        with (logs / f"{name}.log").open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        progress.append({"stage": name, "exit_code": result.returncode, "seconds": time.time() - started})
        progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {logs / (name + '.log')}")
        print(f"DONE {name}: {progress[-1]['seconds']:.1f}s", flush=True)
    print(f"Reproduction saved to {root}", flush=True)


if __name__ == "__main__":
    main()
