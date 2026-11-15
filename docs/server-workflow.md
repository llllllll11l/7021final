# GPU 實驗操作流程

從倉庫根目錄執行。下列命令使用 `configs/default.json`，資料與輸出分別寫入 `artifacts/data` 和 `artifacts/runs`。

## 獨立環境

Python 3.11 以上。RTX 5090 實驗使用 Python 3.12、PyTorch 2.9.1、CUDA 12.8；RTX 5080 實驗使用 Python 3.11、PyTorch 2.9.1、CUDA 13.0。固定套件清單見 `pyproject.toml` 的 `train` extra。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[train,notebook]"
python -m pip check
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 啟用環境，環境變數以 `$env:CUDA_VISIBLE_DEVICES = "0"` 和 `$env:HF_HUB_DISABLE_XET = "1"` 設定。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported()); print((torch.ones(16,16,device='cuda',dtype=torch.bfloat16) @ torch.ones(16,16,device='cuda',dtype=torch.bfloat16)).mean().item())"
```

## 載入既有結果

```bash
python -m gsm8k_lora.artifacts fetch
python -m gsm8k_lora.reproduce --report artifacts/verification.json
```

這兩步下載 RTX 5090 實驗資料並在 CPU 上核對結果。已存在完整本地產物時直接執行第二步。下載目錄須為空，其他位置可用 `--destination` 指定。

## 新增 GPU 實驗

先檢查 GPU 執行鏈路：

```bash
python scripts/check_gpu.py
```

此檢查使用本地 tokenizer、8 筆準備資料和小型隨機初始化的 Qwen2，測試 BF16、loss、兩步 LoRA 訓練、adapter 儲存與生成，結果保存於 `artifacts/gpu-smoke/`。

使用還原後的準備資料，在新目錄重新評分、選樣、B/C 正式訓練，再進行三組完整驗證與測試：

```bash
python scripts/reproduce_gpu.py --output artifacts/reproduction-rtx5080
```

此入口每個階段啟動獨立子程序，將日誌寫到輸出目錄的 `logs/`，完成狀態寫入 `progress.json`。使用 `--train-only` 可先驗證評分與兩次正式訓練。

使用 `--batch-size 2` 可將 microbatch 調為 2、梯度累積調為 16，保持有效批量 32。已有成功階段時可加上 `--resume` 跳過；失敗階段的輸出先移走，再繼續執行。

## 從零準備資料

使用新的 `data_dir` 和 `output_dir`，在設定檔中指定，路徑相對於設定檔所在目錄。

```bash
python -m gsm8k_lora.prepare --config configs/default.json
python -m gsm8k_lora.run --config configs/default.json score
python -m gsm8k_lora.select_data --config configs/default.json
```

檢查 `metadata.json` 的長度分布、`excluded.json` 的排除原因及 `mask_audit.json` 的監督內容。

## 試跑與正式訓練

```bash
python -m gsm8k_lora.run train --method random --seed 13 --debug-steps 10
python -m gsm8k_lora.run eval --method base --split validation

python -m gsm8k_lora.run train --method random --seed 13
python -m gsm8k_lora.run train --method moderate --seed 13
python -m gsm8k_lora.run eval --method random --seed 13 --split validation
python -m gsm8k_lora.run eval --method moderate --seed 13 --split validation
python -m gsm8k_lora.summarize --split validation
```

試跑保存在 `debug_random_13`。正式訓練的 `train.json` 應記錄完整清單一遍與 80 個更新步。`trainer_state.json` 可檢查逐步 loss。

若驗證結果的生成達上限率較高，先統一調整 `generation.max_new_tokens`，再重新驗證三組。正式測試前固定共同設定。

## 完整測試與推理

```bash
python -m gsm8k_lora.run eval --method base --split test
python -m gsm8k_lora.run eval --method random --seed 13 --split test
python -m gsm8k_lora.run eval --method moderate --seed 13 --split test
python -m gsm8k_lora.summarize --split test

python -m gsm8k_lora.run predict --method moderate --seed 13 \
  --question "A shop has 12 apples and sells 5. How many apples remain?"
```

測試使用完整 1,319 題。`summary_test.csv` 保存三組結果；`comparison_test.json` 保存配對區間；`error_cases_test.jsonl` 保存人工分析樣例。

## 失敗恢復

| 現象 | 操作 |
| --- | --- |
| 評分顯存不足 | 降低 `score_batch_size`，移走未完成的 `data/scores/` 後重跑 |
| 訓練顯存不足 | batch 由 4 改成 2，梯度累積由 8 改成 16；必要時啟用 gradient checkpointing，兩組使用相同設定 |
| 生成顯存不足 | 降低 `generation.batch_size`，將該組未完成的 `eval_*` 目錄移走後重跑 |
| 網路下載等待 | 先按 metadata 的模型 SHA 下載；快取完整後可設定 `HF_HUB_OFFLINE=1` |
| 輸出目錄已存在 | 確認該階段是否完成；新實驗改用新的輸出目錄 |

各階段保護現有輸出。成功的評分快取可復用；失敗的訓練或生成階段需要重新執行。

## 產物保存

將各顯卡實驗的 `data/`、`runs/`、`logs/`、`provenance/` 按目錄放到 Hugging Face，保留 checkpoint 和 notebook 暫存檔。RTX 5090 資料位於 `seed13/`，RTX 5080 資料位於 `rtx5080/`。各自的 `results/<實驗>/artifacts.json` 記錄目錄與固定 revision，可使用同一個下載入口取得資料。
