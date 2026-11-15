# RTX 5090 Experiment Results

This directory contains compact records from the RTX 5090 experiment. The training seed is 13 and the data-split seed is 2026.

| File | Contents |
| --- | --- |
| `summary_test.csv` | Metrics, budgets, and costs for all 1,319 test questions |
| `summary_validation.csv` | The same measurements on 500 validation questions |
| `comparison_*.json` | Paired B/C bootstrap comparisons and cost accounting |
| `config.json` | Frozen experiment configuration with its recorded path strings |
| `environment.lock.txt` | Linux / RTX 5090 package snapshot |
| `reproduction.json` | CPU verification of saved data, selections, predictions, and summaries |
| `artifacts.json` | Hugging Face repository, directory, and pinned revision |

Accuracy and rate columns use values from 0 to 1. Time columns use minutes. `T` counts complete non-padding training tokens; `T_y` counts answer tokens included in the loss.

The frozen configuration and environment preserve this experiment's provenance. Start new experiments with the repository's `configs/default.json` and `pyproject.toml`.

Prepared data, scores, selections, adapters, per-question predictions, logs, and notebook checkpoints are available in the [Hugging Face seed13 directory](https://huggingface.co/datasets/tomtom123/7021final-artifacts/tree/main/seed13).
