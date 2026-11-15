"""Check BF16, completion loss, LoRA training, adapter saving, and generation on CUDA."""

import argparse
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsm8k_lora.common import CompletionCollator, environment, load_config, read_jsonl, require_cuda, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "gpu-smoke")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts" / "data")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, set_seed
    from trl import SFTConfig
    from gsm8k_lora.run import generate
    from gsm8k_lora.training import BudgetTrainer

    require_cuda()
    set_seed(13)
    matrix = torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)
    product = float((matrix @ matrix).mean())
    assert product == 16
    tokenizer = AutoTokenizer.from_pretrained(args.data / "tokenizer", padding_side="left")
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=768, tie_word_embeddings=True,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    rows = read_jsonl(args.data / "candidates.jsonl")[:8]
    collator = CompletionCollator(tokenizer.pad_token_id)
    batch = {key: value.to("cuda") for key, value in collator(rows[:2]).items()}
    with torch.no_grad():
        initial_loss = float(model(**batch).loss)
    assert math.isfinite(initial_loss)
    settings = SFTConfig(output_dir=str(args.output), max_steps=2, per_device_train_batch_size=2,
                         gradient_accumulation_steps=1, bf16=True, fp16=False,
                         max_length=768, packing=False, completion_only_loss=True,
                         dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
                         save_strategy="no", eval_strategy="no", logging_steps=1, report_to="none",
                         gradient_checkpointing=False, dataloader_num_workers=0, seed=13)
    trainer = BudgetTrainer(model=model, args=settings, processing_class=tokenizer,
                           train_dataset=Dataset.from_list(rows), data_collator=collator,
                           peft_config=LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    result = trainer.train()
    assert math.isfinite(result.training_loss) and trainer.state.global_step == 2
    trainer.model.save_pretrained(args.output / "adapter")
    trainer.save_state()
    runtime_config = load_config(ROOT / "configs" / "default.json")
    runtime_config["generation"]["max_new_tokens"] = 8
    generated = generate(trainer.model.eval(), tokenizer, ["What is 2 + 3?"], runtime_config)[0]
    assert generated["generated_tokens"] > 0
    report = {"status": "passed", "scope": "Randomly initialized tiny Qwen2 GPU smoke test",
              "environment": environment(), "python": sys.version.split()[0],
              "bf16_matmul_mean": product, "initial_completion_loss": initial_loss,
              "training_loss": result.training_loss, "update_steps": trainer.state.global_step,
              "T": trainer.exposure_tokens, "T_y": trainer.answer_tokens,
              "generated_tokens": generated["generated_tokens"],
              "adapter_saved": (args.output / "adapter" / "adapter_model.safetensors").exists(),
              "model_parameters": sum(parameter.numel() for parameter in trainer.model.parameters())}
    write_json(args.output / "result.json", report)
    print(report)


if __name__ == "__main__":
    main()
