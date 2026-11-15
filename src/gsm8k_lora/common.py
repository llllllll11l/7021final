import hashlib
import importlib.metadata
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path


PREPARATION_KEYS = (
    "model", "model_revision", "dataset", "dataset_revision", "system_prompt",
    "split_seed", "validation_size", "max_length",
)
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+"
FINAL_NUMBER = re.compile(rf"\s*[$£€]?\s*({NUMBER})\s*[$£€]?\s*\.?\s*")
FALLBACK_NUMBER = re.compile(rf"(?<![\w.,/+\-])(?:{NUMBER})(?![\w,/]|\.\d)")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path=DEFAULT_CONFIG):
    """Resolve runtime paths relative to the configuration file."""
    path = Path(path).expanduser().resolve()
    config = read_json(path)
    for key in ("data_dir", "output_dir"):
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent / value).resolve())
    return config


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_prepared(config):
    root = Path(config["data_dir"])
    metadata = read_json(root / "metadata.json")
    revision_keys = ("model_revision", "dataset_revision")
    if any(config[key] != metadata["config"][key] for key in PREPARATION_KEYS if key not in revision_keys):
        raise ValueError("数据相关配置已变化，请使用新的 data_dir 重新 prepare。")
    if any(config[key] not in (metadata["config"][key], metadata[key]) for key in revision_keys):
        raise ValueError("模型或数据版本已变化，请使用新的 data_dir 重新 prepare。")
    return root, metadata


def messages(question, config):
    return [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": question},
    ]


def encode_record(row, tokenizer, config):
    """评分和训练共用完整 chat 序列，只有 assistant 回答及结束标记参与损失。"""
    prompt = messages(row["question"], config)
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": row["answer"]}], tokenize=True,
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template 的 prompt 不是完整序列前缀，不能可靠构造 loss mask。")
    end = len(full_ids)
    while end > len(prompt_ids) and full_ids[end - 1] != tokenizer.eos_token_id:
        end -= 1
    if end <= len(prompt_ids) or tokenizer.decode(full_ids[end:]).strip():
        raise ValueError("回答缺少预期的 EOS，或 EOS 后存在非空内容。")
    full_ids = full_ids[:end]
    return {
        **row, "input_ids": full_ids,
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
        "input_length": len(prompt_ids), "answer_length": len(full_ids) - len(prompt_ids),
        "length": len(full_ids),
    }


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch

        width = max(len(row["input_ids"]) for row in rows)
        inputs, labels, masks = [], [], []
        for row in rows:
            size = len(row["input_ids"])
            inputs.append(row["input_ids"] + [self.pad_token_id] * (width - size))
            labels.append(row["labels"] + [-100] * (width - size))
            masks.append([1] * size + [0] * (width - size))
        return {key: torch.tensor(value, dtype=torch.long) for key, value in
                (("input_ids", inputs), ("labels", labels), ("attention_mask", masks))}


def extract_answer(text, strict=False):
    """最后一个 #### 优先；仅缺标记时回退到末行，拒绝分数等不支持的数值格式。"""
    text = text.replace("−", "-").strip()
    text = re.sub(r"([+-])\s*([$£€])\s*", r"\2\1", text)
    if "####" in text:
        field = text.rsplit("####", 1)[1].strip().splitlines()
        match = FINAL_NUMBER.fullmatch(field[0].strip("* ")) if field else None
        value = match.group(1) if match else None
    elif strict or not text:
        value = None
    else:
        line = text.splitlines()[-1]
        matches = [match for match in FALLBACK_NUMBER.finditer(line)
                   if not re.search(r"/\s*$", line[:match.start()])
                   and not re.match(r"\s*/", line[match.end():])]
        value = matches[-1].group() if matches else None
    return Decimal(value.replace(",", "")) if value is not None else None


def distribution(rows):
    result = {"count": len(rows), "T": sum(row["length"] for row in rows),
              "T_y": sum(row["answer_length"] for row in rows)}
    for key in ("input_length", "answer_length", "length"):
        values = sorted(row[key] for row in rows)
        result[key] = {
            "mean": sum(values) / len(values) if values else 0,
            "quantiles": {str(percent): values[round((len(values) - 1) * percent / 100)] if values else 0
                          for percent in (0, 25, 50, 75, 100)},
            "histogram_128_tokens": {str(edge): sum(edge <= value < edge + 128 for value in values)
                                     for edge in range(0, max(values, default=0) + 1, 128)},
        }
    return result


def require_cuda():
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("本项目仅运行单卡，请使用 python 启动。")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("请设置 CUDA_VISIBLE_DEVICES，只暴露一张 GPU。")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("评分、正式训练和推理需要支持 BF16 的 CUDA GPU。")
    torch.cuda.set_device(0)


def load_model(config, metadata):
    import torch
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    print(f"[model] 加载 {config['model']}，版本 {metadata['model_revision']}（下载/读取缓存）", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=metadata["model_revision"],
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    print(f"[model] 权重读取完成，用时 {time.perf_counter() - started:.1f}s；开始迁移至 GPU", flush=True)
    model = model.to("cuda")
    torch.cuda.synchronize()
    print(f"[model] GPU 就绪，总加载用时 {time.perf_counter() - started:.1f}s", flush=True)
    return model


def start_meter():
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def finish_meter(start):
    import torch

    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


def environment():
    import torch

    return {"packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "huggingface-hub")},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
