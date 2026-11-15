# Seed 13 結果快照

這個目錄保留原始 RTX 5090 實驗的精簡紀錄。訓練種子為 13，資料切分種子為 2026。

| 檔案 | 內容 |
| --- | --- |
| `summary_test.csv` | 完整 1,319 題測試集的三組指標、預算與成本 |
| `summary_validation.csv` | 500 題驗證集的相同指標 |
| `comparison_*.json` | B/C 配對 bootstrap 與成本口徑 |
| `config.json` | 原始凍結設定，保留當時的路徑字串 |
| `environment.lock.txt` | 原始 Linux / RTX 5090 環境的 pip freeze |
| `reproduction.json` | 從保存的資料與預測重新計算的 CPU 核對結果 |
| `artifacts.json` | Hugging Face 下載位置、目錄與固定 revision |

CSV 中 accuracy、strict_accuracy 與各種 rate 使用 0–1 比例，時間使用分鐘。`T` 表示完整非 padding 訓練 token；`T_y` 表示參與 loss 的回答 token。

原始設定與環境檔用於追溯。執行新實驗使用根目錄的 `configs/default.json` 與 `pyproject.toml`。

完整資料、評分、選樣、adapter、逐題預測、日誌與暫存檔見 [Hugging Face 的 seed13 目錄](https://huggingface.co/datasets/tomtom123/7021final-artifacts/tree/main/seed13)。
