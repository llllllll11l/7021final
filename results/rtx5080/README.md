# RTX 5080 Reproduction

The full experiment was rerun on an NVIDIA GeForce RTX 5080 (16 GB) in an isolated Python 3.11.9 environment on Windows. Fresh scoring covered all 6,973 candidates. Both LoRA adapters completed 80 updates, followed by all 500 validation and 1,319 test questions for each of the three methods. CPU verification passed for **all 5,457 saved predictions**, deterministic selection, training budgets, metrics, and paired bootstrap summaries.

## Results

| Method | Validation accuracy | Test accuracy | Test answer format | Mean test tokens | Score + train |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 81.40% | 68.46% | 17.44% | 182.67 | 0.00 s |
| Random LoRA | 73.60% | 62.55% | 98.41% | 93.06 | 155.88 s |
| Moderate-loss LoRA | 71.40% | 62.70% | 98.94% | 92.14 | 292.61 s |

On the test set, C minus B is **+0.15 percentage points**, with a paired bootstrap 95% interval of **[-1.52, +1.74] percentage points**. The interval reflects question-level uncertainty for one training seed. The primary metric uses the same numerical parser as the original run.

Random LoRA consumed **499,800** complete training tokens and 233,576 answer tokens across 2,547 examples. Moderate-loss LoRA consumed **499,848** complete training tokens and 233,886 answer tokens across 2,548 examples. Time columns in the CSV files use minutes; the table above converts scoring and training time to seconds. Downloads and model loading are excluded from these stage timers.

## Execution settings

- PyTorch 2.9.1+cu130, CUDA 13.0, BF16; full package versions are in `environment.lock.txt`.
- Training seed 13, split seed 2026, LoRA rank 16, alpha 32, dropout 0.05, learning rate 0.0001.
- Training microbatch 2 with 16 accumulation steps: effective batch 32. The original run used microbatch 4 with 8 accumulation steps.
- Greedy generation, batch 8, maximum 512 new tokens.
- Model commit: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- GSM8K commit: `740312add88f781978c0658806c59bc2815b9866`.

The rerun reused the original prepared splits and tokenizer, then recomputed candidate losses and both selections. The prepared-data fingerprint matches the original. The frozen `config.json` preserves its recorded symbolic revisions; the actual model and dataset commits above are resolved in the prepared metadata and used for loading.

## Comparison with the original run

The original RTX 5090 test accuracies were 66.94% for Base, 62.17% for Random, and 62.32% for Moderate. This rerun has a different GPU, CUDA build, operating system, Python version, and training microbatch.

Recomputed candidate losses differ from the original by a mean absolute 0.00386415 and maximum 0.03297508. The middle-loss eligibility sets have symmetric differences of 26, 24, 12, 20 IDs across the four length buckets. Both saved training selections differ from the originals; the random condition is length-matched to the moderate condition. These measurements document variation observed between the two execution configurations. The result tables preserve both runs separately.

## Download and verify

From the repository root, with an empty destination:

```bash
python -m pip install -e ".[hub]"
python -m gsm8k_lora.artifacts fetch --source results/rtx5080/artifacts.json --destination artifacts/reproduction-rtx5080
python -m gsm8k_lora.reproduce --artifacts artifacts/reproduction-rtx5080 --report artifacts/rtx5080-verification.json
```

The [Hugging Face rtx5080 folder](https://huggingface.co/datasets/tomtom123/7021final-artifacts/tree/main/rtx5080) contains prepared data, fresh losses and selections, adapters, predictions, logs, stage status, and provenance. `artifacts.json` pins the published revision. `reproduction.json` records the CPU verification result.

To repeat the GPU run after obtaining the original prepared data:

```bash
python scripts/reproduce_gpu.py --output artifacts/my-rerun --batch-size 2
```

Open `notebooks/experiments.ipynb` and set `EXPERIMENT="rtx5080"` to inspect this snapshot. The optional GPU cells use the saved run configuration.
