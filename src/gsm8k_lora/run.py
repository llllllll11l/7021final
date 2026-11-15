import argparse
import math
from pathlib import Path

from .common import (
    DEFAULT_CONFIG, load_config,
    CompletionCollator, environment, extract_answer, fingerprint, finish_meter,
    load_model, load_prepared, messages, read_json, read_jsonl, require_cuda,
    start_meter, write_json, write_jsonl,
)


def score(config):
    import torch
    from transformers import AutoTokenizer

    root, metadata = load_prepared(config)
    output = root / "scores"
    if (output / "metadata.json").exists():
        cached = read_json(output / "metadata.json")
        if cached["data_fingerprint"] != metadata["fingerprint"]:
            raise ValueError("已有评分缓存不匹配当前数据。")
        print(f"复用完整评分缓存：{output}")
        return
    if output.exists():
        raise FileExistsError(f"发现未完成评分目录 {output}，移除后重新评分。")
    require_cuda()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    model = load_model(config, metadata).eval()
    rows = sorted(read_jsonl(root / "candidates.jsonl"), key=lambda row: row["length"])
    collate = CompletionCollator(tokenizer.pad_token_id)
    output.mkdir()
    started = start_meter()
    losses = []
    batch_size = config["score_batch_size"]
    print(f"[score] 开始评分：{len(rows)} 条，batch_size={batch_size}；等待首批计算完成", flush=True)
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset:offset + batch_size]
            batch = {key: value.to("cuda") for key, value in collate(batch_rows).items()}
            labels = batch.pop("labels")[:, 1:]
            logits = model(**batch, use_cache=False).logits[:, :-1, :]
            for index, row in enumerate(batch_rows):
                mask = labels[index] != -100
                loss = torch.nn.functional.cross_entropy(logits[index][mask].float(), labels[index][mask])
                if not torch.isfinite(loss):
                    raise ValueError(f"评分出现非有限损失：{row['id']}")
                losses.append({"id": row["id"], "loss": loss.item(),
                               "answer_tokens": int(mask.sum()), "length": row["length"]})
            del logits
            if offset % (batch_size * 10) == 0 or offset + batch_size >= len(rows):
                print(f"score {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
    cost = finish_meter(started)
    write_jsonl(output / "losses.jsonl", losses)
    write_json(output / "metadata.json", {
        "data_fingerprint": metadata["fingerprint"], "model_revision": metadata["model_revision"],
        "count": len(losses), "cost": cost, "environment": environment(), "config": config,
    })
    print(f"[score] 已保存 {len(losses)} 条损失，用时 {cost['seconds'] / 60:.2f} 分钟：{output}", flush=True)


def train(config, method, seed, debug_steps=0):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from trl import SFTConfig
    from .training import BudgetTrainer

    root, metadata = load_prepared(config)
    selection = read_json(root / "selections" / f"{method}_{seed}.json")
    if selection["data_fingerprint"] != metadata["fingerprint"]:
        raise ValueError("选样清单与当前数据不一致。")
    ids = selection["ids"]
    if len(ids) != len(set(ids)) or fingerprint(ids) != selection["ids_fingerprint"]:
        raise ValueError("选样清单 ID 重复或内容发生变化。")
    candidates = {row["id"]: row for row in read_jsonl(root / "candidates.jsonl")}
    rows = [candidates[identifier] for identifier in ids]
    if any(row["length"] > config["max_length"] for row in rows):
        raise ValueError("选样清单含超长样本；不允许截断训练答案。")
    name = f"{method}_{seed}" if not debug_steps else f"debug_{method}_{seed}"
    output = Path(config["output_dir"]) / name
    if output.exists():
        raise FileExistsError(f"{output} 已存在；不会覆盖已有 adapter。")
    require_cuda()
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    model = load_model(config, metadata)
    settings = config["train"]
    training_args = SFTConfig(
        output_dir=str(output), num_train_epochs=1, max_steps=debug_steps if debug_steps else -1,
        per_device_train_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"], warmup_ratio=settings["warmup_ratio"],
        weight_decay=settings["weight_decay"], lr_scheduler_type="linear", optim="adamw_torch",
        bf16=True, fp16=False, max_length=config["max_length"], packing=False,
        completion_only_loss=True, dataset_kwargs={"skip_prepare_dataset": True},
        group_by_length=True, length_column_name="length", remove_unused_columns=False,
        gradient_checkpointing=settings["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed, data_seed=seed, dataloader_drop_last=False, dataloader_num_workers=0,
        save_strategy="no", eval_strategy="no", logging_steps=1, report_to="none",
    )
    lora = LoraConfig(
        task_type="CAUSAL_LM", r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"], target_modules=settings["target_modules"], bias="none",
    )
    dataset = Dataset.from_list([{key: row[key] for key in ("input_ids", "labels", "length")} for row in rows])
    trainer = BudgetTrainer(
        model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer,
        data_collator=CompletionCollator(tokenizer.pad_token_id), peft_config=lora,
    )
    write_json(output / "config.json", config)
    write_json(output / "selection.json", selection)
    started = start_meter()
    result = trainer.train()
    cost = finish_meter(started)
    actual = {"T": trainer.exposure_tokens, "T_y": trainer.answer_tokens,
              "exposure_examples": trainer.exposure_examples, "update_steps": trainer.state.global_step}
    if not debug_steps:
        expected = {"T": sum(row["length"] for row in rows),
                    "T_y": sum(row["answer_length"] for row in rows), "exposure_examples": len(rows),
                    "update_steps": math.ceil(len(rows) / (settings["batch_size"] * settings["gradient_accumulation_steps"]))}
        if any(actual[key] != value for key, value in expected.items()):
            raise RuntimeError(f"实际训练暴露与清单不一致：actual={actual}, expected={expected}")
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    trainer.save_state()
    write_json(output / "train.json", {
        "method": method, "seed": seed, "debug": bool(debug_steps), "actual": actual,
        "unique_samples": len(set(ids)), "cost": cost, "train_metrics": result.metrics,
        "config": config, "data_fingerprint": metadata["fingerprint"],
        "model_revision": metadata["model_revision"], "environment": environment(),
    })
    print(actual)


def get_inference_model(config, root, metadata, method, seed):
    from peft import PeftModel
    from transformers import AutoTokenizer

    name = "base" if method == "base" else f"{method}_{seed}"
    output = Path(config["output_dir"]) / name
    if method != "base":
        training = read_json(output / "train.json")
        if training["debug"] or training["data_fingerprint"] != metadata["fingerprint"]:
            raise ValueError("Adapter 来自调试运行或不匹配的数据。")
        if training["config"]["train"] != config["train"]:
            raise ValueError("当前训练配置与 adapter 不一致；请恢复冻结的配置。")
    model = load_model(config, metadata)
    if method != "base":
        model = PeftModel.from_pretrained(model, output / "adapter")
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", padding_side="left")
    return model.eval(), tokenizer, output


def generate(model, tokenizer, questions, config):
    import torch
    from transformers import GenerationConfig

    prompts = [tokenizer.apply_chat_template(messages(question, config), tokenize=False, add_generation_prompt=True)
               for question in questions]
    batch = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=config["generation"]["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True,
    )
    with torch.inference_mode():
        outputs = model.generate(**batch, generation_config=generation)
    results = []
    for output in outputs[:, batch["input_ids"].shape[1]:].tolist():
        if tokenizer.eos_token_id in output:
            output = output[:output.index(tokenizer.eos_token_id) + 1]
        results.append({"output": tokenizer.decode(output, skip_special_tokens=True),
                        "generated_tokens": len(output),
                        "hit_limit": len(output) >= generation.max_new_tokens})
    return results


def evaluate(config, method, seed, split):
    from transformers import set_seed

    root, metadata = load_prepared(config)
    output = Path(config["output_dir"]) / ("base" if method == "base" else f"{method}_{seed}") / f"eval_{split}"
    if output.exists():
        raise FileExistsError(f"{output} 已存在；如需调整生成设置，请先保存旧结果。")
    require_cuda()
    set_seed(2026)
    model, tokenizer, _ = get_inference_model(config, root, metadata, method, seed)
    rows = read_jsonl(root / f"{split}.jsonl")
    expected_count = metadata["official_test_count"] if split == "test" else metadata["validation_count"]
    if len(rows) != expected_count or len({row["id"] for row in rows}) != expected_count:
        raise ValueError("评估数据不完整或包含重复 ID。")
    output.mkdir(parents=True)
    started = start_meter()
    predictions = []
    batch_size = config["generation"]["batch_size"]
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        import json

        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset:offset + batch_size]
            generated = generate(model, tokenizer, [row["question"] for row in batch_rows], config)
            for row, result in zip(batch_rows, generated):
                gold = extract_answer(row["answer"], strict=True)
                if gold is None:
                    raise ValueError(f"金标准无效：{row['id']}")
                prediction = extract_answer(result["output"])
                strict_prediction = extract_answer(result["output"], strict=True)
                record = {**row, **result, "gold": str(gold),
                          "prediction": str(prediction) if prediction is not None else None,
                          "strict_prediction": str(strict_prediction) if strict_prediction is not None else None,
                          "correct": prediction == gold, "strict_correct": strict_prediction == gold}
                predictions.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{method} {split} {len(predictions)}/{len(rows)}", flush=True)
    cost = finish_meter(started)
    count = len(rows)
    metrics = {
        "count": count, "accuracy": sum(row["correct"] for row in predictions) / count,
        "strict_accuracy": sum(row["strict_correct"] for row in predictions) / count,
        "strict_format_rate": sum(row["strict_prediction"] is not None for row in predictions) / count,
        "parse_failure_rate": sum(row["prediction"] is None for row in predictions) / count,
        "hit_limit_rate": sum(row["hit_limit"] for row in predictions) / count,
        "mean_generated_tokens": sum(row["generated_tokens"] for row in predictions) / count,
    }
    write_json(output / "metrics.json", {
        **metrics, "method": method, "seed": None if method == "base" else seed,
        "split": split, "cost": cost, "config": config, "environment": environment(),
        "data_fingerprint": metadata["fingerprint"], "model_revision": metadata["model_revision"],
    })
    print(metrics)


def main():
    parser = argparse.ArgumentParser(description="GSM8K 候选评分、单卡 LoRA 训练、评估与命令行推理")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("score")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--method", choices=("random", "moderate"), required=True)
    train_parser.add_argument("--seed", type=int, default=13)
    train_parser.add_argument("--debug-steps", type=int, default=0)
    for command in ("eval", "predict"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--method", choices=("base", "random", "moderate"), required=True)
        subparser.add_argument("--seed", type=int, default=13)
        if command == "eval":
            subparser.add_argument("--split", choices=("validation", "test"), default="validation")
        else:
            subparser.add_argument("--question", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "score":
        score(config)
    elif args.command == "train":
        if args.debug_steps < 0:
            parser.error("--debug-steps 不能为负数")
        train(config, args.method, args.seed, args.debug_steps)
    elif args.command == "eval":
        evaluate(config, args.method, args.seed, args.split)
    else:
        root, metadata = load_prepared(config)
        require_cuda()
        model, tokenizer, _ = get_inference_model(config, root, metadata, args.method, args.seed)
        print(generate(model, tokenizer, [args.question], config)[0]["output"])


if __name__ == "__main__":
    main()
