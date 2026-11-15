import argparse
import random
import re
import time
import unicodedata
from pathlib import Path

from common import distribution, encode_record, extract_answer, fingerprint, read_json, write_json, write_jsonl


def normalize_question(question):
    return " ".join(unicodedata.normalize("NFKC", question).split())


def prepare(config):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    root = Path(config["data_dir"])
    if root.exists():
        raise FileExistsError(f"{root} 已存在；更改 data_dir 可保留现有实验。")
    start = time.perf_counter()
    api = HfApi()
    model_revision = api.model_info(config["model"], revision=config["model_revision"]).sha
    dataset_revision = api.dataset_info(config["dataset"], revision=config["dataset_revision"]).sha
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=model_revision)
    dataset = load_dataset(config["dataset"], "main", revision=dataset_revision)
    groups = {}
    for index, row in enumerate(dataset["train"]):
        key = normalize_question(row["question"])
        groups.setdefault(key, []).append({"id": f"train-{index:05d}", **row})
    unique = []
    duplicates = []
    for group in groups.values():
        unique.append(group[0])
        if len(group) > 1:
            duplicates.append({"kept": group[0]["id"], "removed": [row["id"] for row in group[1:]],
                               "conflicting_answers": len({row["answer"] for row in group}) > 1})
    random.Random(config["split_seed"]).shuffle(unique)
    size = config["validation_size"]
    if not 0 < size < len(unique):
        raise ValueError("validation_size 必须小于去重后的训练集大小。")
    validation = unique[:size]
    candidates, excluded, lengths = [], [], []
    for row in unique[size:]:
        if not row["question"].strip() or extract_answer(row["answer"], strict=True) is None:
            excluded.append({"id": row["id"], "reason": "invalid_question_or_gold"})
            continue
        cleaned = {**row, "answer": re.sub(r"<<.*?>>", "", row["answer"], flags=re.DOTALL).strip()}
        encoded = encode_record(cleaned, tokenizer, config)
        lengths.append(encoded)
        if encoded["length"] > config["max_length"]:
            excluded.append({"id": row["id"], "reason": "too_long", "length": encoded["length"]})
        else:
            candidates.append(encoded)
    test = [{"id": f"test-{index:05d}", **row} for index, row in enumerate(dataset["test"])]
    for row in validation + test:
        if extract_answer(row["answer"], strict=True) is None:
            raise ValueError(f"参考答案无法解析：{row['id']}；不允许丢弃验证/测试题。")
    if not candidates:
        raise ValueError("候选池为空，请检查最长序列配置。")
    root.mkdir(parents=True)
    tokenizer.save_pretrained(root / "tokenizer")
    write_jsonl(root / "candidates.jsonl", candidates)
    write_jsonl(root / "validation.jsonl", validation)
    write_jsonl(root / "test.jsonl", test)
    write_json(root / "duplicates.json", duplicates)
    write_json(root / "excluded.json", excluded)
    metadata = {
        "config": config, "model_revision": model_revision, "dataset_revision": dataset_revision,
        "fingerprint": fingerprint({"candidates": candidates, "validation": validation, "test": test,
                                    "model_revision": model_revision, "dataset_revision": dataset_revision}),
        "official_train_count": len(dataset["train"]), "official_test_count": len(test),
        "validation_count": len(validation), "duplicate_removed": sum(len(row["removed"]) for row in duplicates),
        "excluded_count": len(excluded), "before_length_filter": distribution(lengths),
        "candidates": distribution(candidates), "prepare_wall_seconds": time.perf_counter() - start,
    }
    write_json(root / "metadata.json", metadata)
    audit = []
    for row in candidates[:3]:
        audit.append({"id": row["id"], "input_length": row["input_length"],
                      "answer_length": row["answer_length"],
                      "masked_prompt": tokenizer.decode(row["input_ids"][:row["input_length"]]),
                      "supervised_completion": tokenizer.decode(row["input_ids"][row["input_length"]:])})
    write_json(root / "mask_audit.json", audit)
    print(f"候选 {len(candidates)}，验证 {len(validation)}，测试 {len(test)}，过滤 {len(excluded)}。")
    print(f"检查 {root / 'metadata.json'} 与 mask_audit.json 后再评分。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备 GSM8K、冻结版本并构造 completion loss mask")
    parser.add_argument("--config", default="config.json")
    prepare(read_json(parser.parse_args().config))
