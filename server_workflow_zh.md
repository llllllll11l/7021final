# 服务器实验操作流程：仅种子 13

## 当前状态与实验范围

- 环境已经配置完成，服务器项目目录为 `/root/autodl-tmp`。
- 已启用 AutoDL 学术资源加速，没有使用镜像。
- 已关闭 Xet，`prepare.py` 已成功完成；不要重新执行 Prepare，也不要删除已有数据。
- 仅使用训练种子 13：A 原始模型、B 分桶随机 SFT、C 中等损失 SFT。共两次正式训练、三次完整测试评估。
- 数据划分种子保持 2026。只修改训练种子列表不需要重新准备数据。

以下命令在服务器执行。按阶段逐块运行，报错时先处理该阶段，不要继续执行后续步骤。

## 1. 恢复终端环境并设置单种子

每次新开终端或重连后，执行：

```bash
cd /root/autodl-tmp
source .venv/bin/activate
source /etc/network_turbo

export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs
set -o pipefail
```

以上环境变量只影响当前终端及其子进程。后续评分会下载完整模型权重，必须继续保持学术资源加速和关闭 Xet 的设置。无需重装依赖、登录 Hugging Face 或切换镜像。

本地 `config.json` 已改成只用 13；服务器上的文件不会自动同步。执行下面的命令，只更新服务器配置中的种子列表，保留其他配置：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("config.json")
config = json.loads(path.read_text())
config["seeds"] = [13]
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
print("正式训练种子:", config["seeds"])
print("数据划分种子:", config["split_seed"])
PY
```

确认 Prepare 的结果和关闭 Xet 的设置：

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import constants
from common import load_prepared, read_json

config = read_json("config.json")
root, metadata = load_prepared(config)
assert config["seeds"] == [13]
assert constants.HF_HUB_DISABLE_XET
for name in ("candidates.jsonl", "validation.jsonl", "test.jsonl"):
    assert (root / name).is_file(), name
print("关闭 Xet:", constants.HF_HUB_DISABLE_XET)
print("候选池:", metadata["candidates"]["count"])
print("验证集:", metadata["validation_count"])
print("测试集:", metadata["official_test_count"])
print("过滤样本:", metadata["excluded_count"])
print("模型版本:", metadata["model_revision"])
PY
```

默认数据预计为候选池 6,973、验证集 500、测试集 1,319；具体以已保存的元数据为准。不要修改 `artifacts/data/metadata.json` 中记录的历史配置。

建议在 `tmux` 中运行较长任务。如果新建了 tmux 会话，应在会话内部执行本节环境设置。离开会话按 `Ctrl+B` 后按 `D`，之后用 `tmux attach -t gsm8k` 恢复。

```bash
tmux new -s gsm8k
```

## 2. 候选池损失评分

先单独下载 Prepare 锁定版本的模型，让下载和 GPU 计算分开。如果旧的 `score` 进程已经长时间停在模型加载阶段，先在其终端按 `Ctrl+C` 并确认退出，不要同时启动两个评分进程。

```bash
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
unset HF_HUB_OFFLINE

python -u - <<'PY' 2>&1 | tee logs/download_model.log
from huggingface_hub import snapshot_download
from common import load_prepared, read_json

config = read_json("config.json")
_, metadata = load_prepared(config)
print("下载模型:", config["model"], flush=True)
print("固定版本:", metadata["model_revision"], flush=True)
path = snapshot_download(
    repo_id=config["model"],
    revision=metadata["model_revision"],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    max_workers=1,
    etag_timeout=60,
)
print("模型下载完成:", path, flush=True)
PY
```

看到“模型下载完成”且命令成功退出后，再开启离线模式评分：

```bash
export HF_HUB_OFFLINE=1
python -u run.py score 2>&1 | tee logs/score.log
```

`snapshot_download` 写入 Hugging Face 缓存，复用已有下载；没有更换模型、版本或缓存目录。下载成功后，评分、训练和评估均可从缓存运行，离线模式避免自动探测 `adapter_config.json` 等可选文件时再次等待网络。后续新终端可在第 1 节设置后再次执行 `export HF_HUB_OFFLINE=1`，但只有模型已完整下载后才能这样做；将来若重新 Prepare 或下载其他模型，应先 `unset HF_HUB_OFFLINE`。

下载阶段失败时不要继续评分，也不要删除 Hugging Face 缓存。主站 API 可访问不代表文件存储域名一定可访问；如仍超时，保留 `logs/download_model.log` 中的具体失败地址用于排查。

评分在 GPU 上计算每条候选答案的平均损失，日志会输出 `score .../6973`。此阶段没有反向传播，不训练 adapter。更新后的 `common.py` 和 `run.py` 还会打印模型加载开始、权重读取完成、GPU 就绪、首批评分开始及评分保存完成；本地修改需要上传到服务器才生效。

完成后检查：

```bash
python - <<'PY'
from common import read_json

result = read_json("artifacts/data/scores/metadata.json")
print("已评分样本:", result["count"])
print("评分分钟:", round(result["cost"]["seconds"] / 60, 2))
print("峰值已分配显存 GiB:", round(result["cost"]["peak_allocated_gib"], 2))
PY
```

成功产物是 `artifacts/data/scores/losses.jsonl` 和 `metadata.json`。完整评分缓存会自动复用，不需要分别为 B/C 评分。

若使用本地模型缓存：

```bash
cd /root/autodl-tmp
source .venv/bin/activate

mkdir -p /root/autodl-tmp/hf-cache
tar -xzf qwen-model-cache.tar.gz -C /root/autodl-tmp/hf-cache

export HF_HUB_CACHE=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs
set -o pipefail

python -u run.py score 2>&1 | tee logs/score.log
```

## 3. 生成种子 13 的 B/C 清单

```bash
python -u select_data.py 2>&1 | tee logs/select.log
```

应生成 `random_13.json`、`moderate_13.json` 和选样元数据。查看实际预算：

```bash
python - <<'PY'
from common import read_json

for method in ("random", "moderate"):
    result = read_json(f"artifacts/data/selections/{method}_13.json")
    summary = result["summary"]
    print(method, {
        "样本数": summary["count"],
        "T": summary["T"],
        "T_y": summary["T_y"],
        "预期更新步数": result["expected_update_steps"],
    })
print("两组相对差异:", result["pair_relative_gaps"])
PY
```

T 是实际完整输入和回答 token 总数，T_y 是参与损失的回答 token 总数。目标预算约 500,000，可能因中等损失池容量统一下调，也可能因完整样本拼装略有不足。脚本要求 B/C 的 T 差异不超过 1%，T_y 和样本数差异不超过 5%。

若之前已经用旧配置生成过 13、42 的清单，可以保留现有清单并直接使用其中的 13，跳过本节生成命令；汇总只读取配置中的种子 13。不要覆盖已经用于训练的清单。

## 4. 短试跑，确认训练显存和流程

先做 10 个更新步的运行检查：

```bash
python -u run.py train --method random --seed 13 --debug-steps 10 \
  2>&1 | tee logs/debug_random_13.log
```

这次运行保存到 `artifacts/runs/debug_random_13/`，不会计入正式实验。10 步只检查能否训练和保存；若要按原方案做更长的性能试跑，可以在首次运行前把 10 改为 100。不要把调试 adapter 当作正式 B。

```bash
python - <<'PY'
from common import read_json

result = read_json("artifacts/runs/debug_random_13/train.json")
print("调试运行:", result["debug"])
print("实际训练暴露:", result["actual"])
print("运行成本:", result["cost"])
PY
```

在另一个终端可以用 `watch -n 2 nvidia-smi` 查看显存。

如果训练 OOM，在 `config.json` 的 `train` 中把 `batch_size` 改为 2、`gradient_accumulation_steps` 改为 16，有效批量仍约 32；必要时把 `gradient_checkpointing` 改为 `true`。将失败的调试目录移走后再试。仅改变这些训练设置，不需要重新 Prepare、评分或选样。

清单中的 `expected_update_steps` 是选样时的估计；若之后改变有效批量，以正式训练的 `train.json` 实际步数为准。

## 5. 原始模型验证

在训练设置确认可用后运行：

```bash
python -u run.py eval --method base --split validation \
  2>&1 | tee logs/base_validation.log
```

```bash
python - <<'PY'
from common import read_json

result = read_json("artifacts/runs/base/eval_validation/metrics.json")
for key in ("count", "accuracy", "strict_accuracy", "strict_format_rate",
            "parse_failure_rate", "hit_limit_rate", "mean_generated_tokens"):
    print(key, result[key])
PY
```

各比例范围为 0–1。`hit_limit_rate` 为触及生成上限的比例；若明显高于 0.02，在正式测试前统一将 `generation.max_new_tokens` 从 512 改为 768，移走旧验证评估目录后重跑。若推理 OOM，降低 `generation.batch_size`，例如从 8 改为 4。

本节只做验证，不运行官方测试集。

## 6. 正式训练 B/C，并验证

以下命令块顺序训练随机组和中等损失组，各用种子 13；任一步失败都会停止本命令块：

```bash
(
  set -euo pipefail

  for method in random moderate; do
    python -u run.py train --method "$method" --seed 13 \
      2>&1 | tee "logs/train_${method}_13.log"

    python -u run.py eval --method "$method" --seed 13 --split validation \
      2>&1 | tee "logs/validation_${method}_13.log"
  done

  python -u summarize.py --split validation \
    2>&1 | tee logs/summary_validation.log
)
```

正式命令不带 `--debug-steps`。每组从同一个原始 checkpoint 独立初始化 LoRA，训练一遍各自清单，结束时检查实际 token 和样本暴露。正式训练可能不到 100 步，这是正常情况。

两组 adapter 分别位于：

```text
artifacts/runs/random_13/adapter/
artifacts/runs/moderate_13/adapter/
```

核对实际训练预算和验证结果：

```bash
python - <<'PY'
from common import read_json

for method in ("random", "moderate"):
    result = read_json(f"artifacts/runs/{method}_13/train.json")
    print(method, result["actual"], result["cost"])
PY

cat artifacts/runs/summary_validation.csv
python -m json.tool artifacts/runs/comparison_validation.json
```

验证完成后检查三组的生成上限触及率。如果需要修改生成设置，应统一调整并重新评估三组验证集；仅修改生成设置无需重新训练。如果改变训练超参数，则两组必须使用共同的新配置重新训练，并重跑原始模型验证以保持汇总配置记录一致。

只有种子 13 时，汇总中的 `mean_accuracy` 等于该种子的准确率，不是跨种子平均。

## 7. 冻结配置，完成完整测试

确认不再根据验证集调整配置后保存快照：

```bash
cp config.json config.frozen.json
python -m pip freeze > environment.lock.txt
```

运行 A、B、C 三组完整测试：

```bash
(
  set -euo pipefail

  python -u run.py --config config.frozen.json eval --method base --split test \
    2>&1 | tee logs/test_base.log

  for method in random moderate; do
    python -u run.py --config config.frozen.json eval \
      --method "$method" --seed 13 --split test \
      2>&1 | tee "logs/test_${method}_13.log"
  done

  python -u summarize.py --config config.frozen.json --split test \
    2>&1 | tee logs/summary_test.log
)
```

注意 `run.py` 的 `--config` 放在 `eval` 子命令之前。测试时不再按分数修改参数，每组都应评估完整 1,319 道题。解析失败和生成截断仍计入分母。

## 8. 查看结果、人工分析和演示

```bash
cat artifacts/runs/summary_test.csv
python -m json.tool artifacts/runs/comparison_test.json
```

| 产物 | 用途 |
| --- | --- |
| `artifacts/runs/summary_test.csv` | 三组准确率、格式指标、实际预算、耗时和显存 |
| `artifacts/runs/comparison_test.json` | 种子 13 的 C−B 百分点差、配对 bootstrap 区间 |
| `artifacts/runs/error_cases_test.jsonl` | 三类错误各最多 8 道，共最多 24 道人工分析样例 |
| `artifacts/runs/base/eval_test/predictions.jsonl` | 原始模型全部逐题输出 |
| `artifacts/runs/random_13/eval_test/predictions.jsonl` | 随机组全部逐题输出 |
| `artifacts/runs/moderate_13/eval_test/predictions.jsonl` | 中等损失组全部逐题输出 |

准确率为 0–1 比例；`C_minus_B_pp` 为百分点差。表内 C 的评分时间包含一次完整候选池评分，B 的评分时间为零；实际正式实验成本对共享评分只计一次。下载加载、数据准备、试跑、验证、排障和人时需另行记录，不能把汇总时间称为整个项目总耗时。

人工查看错误样例后，填写 `human_error_type` 和 `human_notes`。可以标记条件理解、推理列式、算术、格式提取或输出不完整。单个种子的结果只能作为初步观察；配对 bootstrap 反映题目采样不确定性，不证明训练种子稳定性。

命令行展示同一道题的三组回答：

```bash
for method in base random moderate; do
  printf '\n=== %s ===\n' "$method"
  python run.py --config config.frozen.json predict \
    --method "$method" --seed 13 \
    --question 'A shop has 12 apples and sells 5. How many apples remain?'
done
```

## 9. 保存与下载实验产物

```bash
tar -czf experiment-seed13-results.tar.gz \
  config.frozen.json environment.lock.txt requirements.txt \
  common.py prepare.py select_data.py run.py summarize.py \
  logs artifacts
```

通过服务器文件面板下载压缩包。包内包含清单、损失缓存、adapter、预测、配置和日志；基础模型通常在 Hugging Face 缓存中，不包含在包内，可按元数据保存的 commit SHA 重新获取。调试目录也保留，以便区分调试开销和正式实验开销。

## 失败后如何继续

代码拒绝覆盖已有结果。完整评分缓存可复用，其他阶段重跑前应先移动该阶段的旧目录。没有断点续训或断点评估，失败阶段需要从头运行，成功的其他阶段保留。

| 情况 | 操作 |
| --- | --- |
| 评分 OOM | 减小 `score_batch_size`，移动未完成的 `artifacts/data/scores/`，重跑评分 |
| 训练 OOM | 减小 `train.batch_size` 并提高梯度累积，必要时开启 checkpointing；移动失败训练目录再跑 |
| 推理 OOM | 减小 `generation.batch_size`，移动失败的 `eval_validation/` 或 `eval_test/` 后再跑 |
| 评估中断 | 只移动对应评估子目录，不要移动或删除已完成 adapter |
| 新终端再次出现网络/Xet 错误 | 在该终端重新执行学术加速与 `export HF_HUB_DISABLE_XET=1` |
| GPU 利用率为 0、只占少量显存、`scores` 目录尚未创建 | 尚未进入评分循环；按第 2 节将模型下载与离线评分分开，定位下载或 CPU 加载等待 |
| 提示目录已存在 | 先判断是否已成功；成功就跳过，失败才移动并重跑 |

例如移动一次失败的调试训练目录：

```bash
mv artifacts/runs/debug_random_13 \
   "artifacts/runs/debug_random_13_previous_$(date +%Y%m%d_%H%M%S)"
```

例如移动原始模型的一次旧验证结果：

```bash
mv artifacts/runs/base/eval_validation \
   "artifacts/runs/base/eval_validation_previous_$(date +%Y%m%d_%H%M%S)"
```

只有对应目录确实存在且需要重跑时才执行移动。改变共同训练配置后，B/C 都需要使用该配置重训；改变共同生成配置后，三组都需使用该配置重新评估。循环中途失败，恢复后从失败的具体命令继续，不要重新执行已成功的训练命令。
