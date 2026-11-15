# GSM8K 固定预算 LoRA 实验

按 `project_plan_zh.md` 实现三组：A 原始 Qwen2.5-1.5B-Instruct、B 长度分桶随机 SFT、C 桶内中等损失 SFT。仅单卡 BF16 LoRA、原生 SDPA；不含网页、额外实验组或服务框架。

当前实验只用种子 13，共两次正式训练。服务器已完成 Prepare、使用 AutoDL 加速并关闭 Xet 时，直接按 [服务器操作流程](server_workflow_zh.md) 从评分继续，无需重新准备数据。

## 1. 服务器安装

建议 Linux、Python 3.12、单张 RTX 5090。先用 `nvidia-smi` 检查驱动，再安装 CUDA 12.8 wheel；无需自行编译 FlashAttention。PyTorch 已从 2.7 开始提供 Blackwell / CUDA 12.8 支持，见[官方说明](https://pytorch.org/blog/pytorch-2-7/)。本项目固定使用 2.9.1，安装命令可核对[官方版本页](https://pytorch.org/get-started/previous-versions/)。

```bash
cd /path/to/Course\ Project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
export CUDA_VISIBLE_DEVICES=0
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print((torch.ones(16,16,device="cuda",dtype=torch.bfloat16) @ torch.ones(16,16,device="cuda",dtype=torch.bfloat16)).mean())'
```

首次执行需要访问 Hugging Face 下载模型与 GSM8K。成功后运行 `python -m pip freeze > environment.lock.txt` 保存服务器完整环境。固定依赖已在本地通过导入和微型模型 CPU 检查；5090 显存、速度及完整训练仍需服务器实测。

## 2. 数据、评分和选样

所有命令从项目根目录运行。默认配置在 `config.json`，路径相对于当前目录。

```bash
python prepare.py
python run.py score
python select_data.py
```

先查看 `artifacts/data/metadata.json` 的长度统计、`excluded.json` 的过滤原因和 `mask_audit.json` 的掩码样例，再进行正式训练。若超长比例不可忽略，统一把 `max_length` 改为 1024，使用新的 `data_dir` 重新准备、评分和选样。

- 官方 train 按规范化题目去重，同题只保留最早一条，保留被删除 ID 和答案冲突信息；再固定抽出 500 条验证题。
- 候选答案移除 `<<...>>`，使用同一 chat template 分词；只去掉 assistant EOS 后的模板换行。系统/用户和 padding 的标签均为 `-100`，回答及 EOS 参与损失。
- 评分为冻结原模型的回答 token 平均负对数似然，无反向传播、无预热。缓存只包含候选池；验证/测试参考答案不进入评分。
- 完整长度分四个分位桶，相同边界合并，小于 8 条的桶合并。C 取每桶损失排名的中间 50%（下界向下、上界向上取整），再随机取样。
- 共同预算按 C 的最小桶容量自动下调；桶配额按完整候选池 token 占比分配。完整样本只使用一次，不截断或重复补足配额。
- 为匹配长度，B 每桶尝试 64 个随机排列，仅以 T、T_y 和样本数差异选择最接近 C 的清单。此步骤不使用 B 的损失值；它是带长度匹配的随机基线。
- 两组 T 差异超过 1%，或 T_y / 样本数差异超过 5%，选样会失败，需检查分布和共同预算。清单同时记录桶直方图、分位数、配额及预期更新步数。

`prepare` 将模型和数据的 `main` 解析成固定 commit SHA，保存 tokenizer；后续阶段使用该版本。准备参数变化会要求重新准备数据。仅将 `seeds` 改为 `[13]` 不影响已准备数据和评分缓存。

## 3. 试跑、验证和正式训练

先检查原始模型在验证集的生成上限，再用 100 步训练检查速度和显存。试跑可重复暴露数据，输出位于独立 `debug_*` 目录，不参与正式比较。

```bash
python run.py eval --method base --split validation
python run.py train --method random --seed 13 --debug-steps 100
```

若 OOM，优先将 `train.batch_size` 从 4 改为 2，同时将 `gradient_accumulation_steps` 从 8 改为 16；必要时统一开启 `gradient_checkpointing`。若验证集 `hit_limit_rate` 超过约 2%，在正式测试前统一提高 `generation.max_new_tokens` 至 768，并重新检查验证结果。修改后的训练、生成设置必须对所有组一致。

正式运行固定一遍完整清单，各自从相同原始模型新建 LoRA adapter：

```bash
for seed in 13; do
  for method in random moderate; do
    python run.py train --method "$method" --seed "$seed" || exit 1
    python run.py eval --method "$method" --seed "$seed" --split validation || exit 1
  done
done
python summarize.py --split validation
```

当前 `config.json` 的 `seeds` 为 `[13]`，B/C 各训练一次。数据划分种子仍固定为 2026。单训练种子的结果属于初步观察，不代表跨种子稳定收益。学习率为 3% warm-up 后线性下降；不按验证或测试表现挑 checkpoint，只保存预算末端 adapter。

每次训练保存实际进入 microbatch 的非 padding T、参与损失的 T_y、样本暴露数和更新步数；正式训练结束会检查它们与清单一致。Trainer 的 `trainer_state.json` 记录逐步损失。若调整超参数，保存旧试验目录，再让 B/C 使用相同新配置重新训练。

## 4. 冻结配置后评估完整测试集

```bash
python run.py eval --method base --split test
for seed in 13; do
  for method in random moderate; do
    python run.py eval --method "$method" --seed "$seed" --split test || exit 1
  done
done
python summarize.py --split test
```

评估不提供测试集抽样或截断选项。固定零样例提示、贪心生成、同一 EOS 和生成上限，只解析新生成的 assistant 内容。解析失败与达到生成上限的题目均保留在分母中。

数值规则：最后一个 `####` 后第一行必须是数值字段（允许货币符号、千位逗号、负号、小数和句末句点）；存在标记但字段无效时计错。仅无标记时回退到最后一个非空行的最后一个合法数值。不计算分数、科学记数或表达式，不四舍五入，用 `Decimal` 精确比较。

主要产物：

| 文件 | 内容 |
| --- | --- |
| `artifacts/data/metadata.json` | 版本、划分、清洗及长度统计 |
| `artifacts/data/{candidates,validation,test}.jsonl` | 稳定 ID 数据及训练 token / mask |
| `artifacts/data/scores/` | 逐样本损失与评分成本 |
| `artifacts/data/selections/` | 每组每种子的 ID 清单和预算 |
| `artifacts/runs/{method}_{seed}/adapter/` | LoRA adapter 与 tokenizer |
| `artifacts/runs/{method}_{seed}/train.json` | 实际训练预算、成本、配置、环境 |
| `artifacts/runs/*/eval_{split}/` | 原始预测、逐题正误、解析和截断指标 |
| `artifacts/runs/summary_test.csv` | 主结果表，准确率为 0–1 比例 |
| `artifacts/runs/comparison_test.json` | 每种子 C−B 百分点差、均值和配对 bootstrap 区间 |
| `artifacts/runs/error_cases_test.jsonl` | 供人工分类的约 24 道三组输出，类别不足则更少 |

汇总会核对各组配置与分母。Bootstrap 只描述题目采样不确定性，不能替代训练种子重复。错误样例按固定种子从共同失败、B 对 C 错、C 对 B 错中抽取，`human_error_type` 和 `human_notes` 留给人工填写。

成本是 CUDA 同步后的阶段墙钟时间与峰值显存，不是 profiler 的纯 kernel 时间。加载/下载不含在评分、训练和生成时间内；数据准备墙钟时间单列。每个 C 的单次方法成本包含一次完整评分；`recorded_formal_compute_minutes` 对共享缓存只计一次，另加全部正式训练和当前 split 评估。选样 CPU 耗时见 selections 元数据；数据准备人时、验证、试跑、排障和租卡费用需另行记录。

## 5. 命令行演示

```bash
for method in base random moderate; do
  python run.py predict --method "$method" --seed 13 \
    --question 'A shop has 12 apples and sells 5. How many apples remain?'
done
```

每个阶段拒绝覆盖已有结果。失败时保留未完成目录，检查原因后移动或删除该阶段目录再重跑；评分已成功时会复用缓存。需要新一轮独立实验可修改 `output_dir`，数据相关设置变化则同时修改 `data_dir`。不实现断点续训，正式单次训练较短，避免混淆预算和 adapter 来源。

主要实现入口：`prepare.py` 数据准备，`select_data.py` 选样，`run.py` 评分/训练/评估/推理，`summarize.py` 结果汇总，`common.py` 共用模板、掩码与答案提取。TRL 使用[预分词与 completion loss 接口](https://huggingface.co/docs/trl/v0.27.0/en/sft_trainer)，显式构造 labels，避免依赖模板的 assistant mask 支持。
