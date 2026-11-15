# 方法與評估

## 任務與三組對照

模型接收一道英文數學應用題，輸出解題步驟及最後一行的 `#### <number>`。A 使用原始 Qwen2.5-1.5B-Instruct；B 和 C 各從相同原始權重新建 LoRA adapter。B 使用長度分桶隨機資料，C 使用同桶內中等損失資料。

三組共享提示、模型版本、tokenizer、EOS、貪心解碼與生成上限。B/C 共享訓練設定與 token 預算。

## 資料準備

來源為 GSM8K `main` 配置，官方 train 7,473 題、test 1,319 題。

1. 使用 NFKC 與空白規範化檢查訓練題目，同題保留最早一條，記錄重複 ID 與答案衝突。
2. 以種子 2026 打散去重後的訓練資料，固定抽出 500 題驗證集。
3. 清除候選答案中的 `<<...>>` 計算註記，保留其餘解題步驟與最終數值。
4. 套用相同 chat template，完整序列最多 768 token。超長或無效的候選樣本記錄於 `excluded.json`。
5. 為候選、驗證與測試資料保留穩定 ID；測試集完整保留。

原始執行產生 6,973 個候選樣本，重複與排除數均為 0。

`prepare` 將模型與資料集的 revision 解析為 commit SHA，保存 tokenizer 與內容 fingerprint。後續階段核對準備參數及 fingerprint。

## Loss mask 與候選評分

系統提示、使用者輸入與 padding 的 label 設為 `-100`。assistant 解答及 EOS 參與 loss，EOS 後的模板空白移除。`mask_audit.json` 保存三個可直接閱讀的範例。

評分使用凍結的原始模型，對每個候選解答計算回答 token 的平均負對數似然：

\[
d_i = -\frac{1}{|y_i|}\sum_t \log p_{\theta_0}(y_{i,t}\mid x_i,y_{i,<t}).
\]

損失描述模型對參考解答的可預測程度，受題目、措辭、格式與原模型知識影響。逐題損失與成本保存於 `data/scores/`。候選池以外的參考解答僅用於評估。

## 長度分桶與預算

完整序列長度按四個分位區間分桶；重複邊界合併，少於 8 筆的桶併入鄰桶。原始執行的桶邊界為 91–155、156–185、186–227、228–488 token。

C 在各桶依 `(loss, id)` 排序，取排名中間 50% 作為合格池，下界向下、上界向上取整。共同預算由 C 各桶容量與完整候選池的 token 權重決定，上限為設定的 500,000。各桶配額按候選池 token 占比分配，完整樣本各使用一次。

C 先依配額抽樣；B 在各桶嘗試 64 個隨機排列，按完整 token、回答 token 和樣本數與 C 的相對差選擇最接近的清單。B 的匹配距離只使用長度與數量。

選樣要求兩組完整 token 相對差不超過 1%，回答 token 與樣本數相對差不超過 5%。清單保存每個 ID、桶分布、指紋、預算和預期步數。

## LoRA 訓練

| 參數 | 設定 |
| --- | --- |
| 精度與 attention | BF16、原生 SDPA |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| 目標模組 | q_proj、k_proj、v_proj、o_proj |
| batch / gradient accumulation | 4 / 8，有效批量 32 |
| 學習率 | 1e-4 |
| 排程 | 3% warm-up，線性下降 |
| 曝露次數 | 完整清單一遍 |
| 訓練種子 | 13 |

每個 microbatch 記錄實際非 padding token `T`、回答 token `T_y`、樣本曝露與更新步。正式訓練結束後，與選樣清單逐項核對。`trainer_state.json` 保存每步 loss，預算末端 adapter 保存於 `runs/{method}_{seed}/adapter/`。

`--debug-steps` 產物使用 `debug_*` 目錄，供速度和顯存檢查。

## 生成與數值判分

使用零樣例提示、greedy decoding、512 個新 token 上限與共同 EOS。只解析新生成的 assistant 內容。

- 有 `####` 時，解析最後一個標記後的首行數值欄位，接受貨幣符號、千位逗號、正負號、小數與句末句點。
- 缺少標記時，主指標回退到最後一個非空行中的最後一個合法數值。
- 嚴格指標只接受標記欄位。分數、科學記數法與未求值表達式依固定規則處理。
- 使用 `Decimal` 精確比較。解析失敗與到達生成上限的題目均計入完整分母。

主要指標為主準確率，並列嚴格準確率、格式合法率、解析失敗率、達上限率與平均生成 token。

## 配對比較與成本

以相同題目 ID 比較 B/C，使用固定 RNG 進行 2,000 次題目配對 bootstrap。區間描述這一對固定模型的題目抽樣不確定性。跨訓練種子的變異需另行重複實驗。

評分、訓練與生成成本使用 CUDA 同步後的牆鐘時間，另記峰值 allocated / reserved 顯存。模型下載與載入時間由日誌記錄。表中每個 C 的方法成本包含一次完整評分；實際總計對共享評分快取計一次。

## 產物

完整檔案由 [Hugging Face](https://huggingface.co/datasets/tomtom123/7021final-artifacts) 封存，本地還原於 `artifacts/`。

| 路徑 | 內容 |
| --- | --- |
| `data/metadata.json` | 原始設定、版本、切分與長度統計 |
| `data/{candidates,validation,test}.jsonl` | 帶穩定 ID 的資料 |
| `data/scores/` | 平均損失、評分成本與環境 |
| `data/selections/` | 每組 ID 清單、分布與 token 預算 |
| `runs/{method}_{seed}/train.json` | 實際訓練預算、成本與設定 |
| `runs/*/eval_{split}/` | 逐題預測與指標 |
| `runs/summary_*.csv` | 三組結果表 |
| `runs/comparison_*.json` | 配對區間與成本總計 |
| `runs/error_cases_*.jsonl` | 固定抽取的人工分析案例 |
| `logs/` | 原始執行日誌與暫存檔 |
