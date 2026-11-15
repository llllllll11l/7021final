# 專案方案

**Moderate-Loss Data Selection for Budget-Constrained LoRA Fine-Tuning on GSM8K**

本方案固定研究範圍、資料選樣和評估方式。種子 13 的已完成結果見 [實驗分析](experiment-analysis.md)。

## 研究問題

1. 約 0.5M 訓練 token 下，LoRA SFT 如何改變 Qwen2.5-1.5B-Instruct 的 GSM8K 準確率與輸出格式？
2. 匹配 token 預算與長度分布後，中等損失選樣相對隨機選樣有何差異？
3. 計入候選池評分後，選樣策略的效果與成本如何權衡？

中等損失策略的動機是，在有限適配預算下使用原模型具有一定熟悉度、同時保留學習空間的參考解答。實驗直接檢驗這個策略在固定設定下的表現。

## 固定實驗矩陣

| 組別 | 模型與資料 | 訓練 |
| --- | --- | --- |
| A | 原始 Qwen2.5-1.5B-Instruct | 直接評估 |
| B | 長度分桶、長度匹配的隨機清單 | 一次獨立 LoRA SFT |
| C | 各長度桶中間 50% 損失池抽樣 | 一次獨立 LoRA SFT |

資料切分種子固定為 2026，最小實驗使用訓練種子 13。增加實驗時優先重複 B/C 的訓練種子，保留相同三組定義。

## 系統與資料

輸入為英文數學應用題，輸出為簡潔推理與 `#### <number>`。單卡 BF16 LoRA、SDPA、固定零樣例提示、貪心生成。

GSM8K 官方 train 去重後固定抽出 500 題驗證集，其餘題目作為候選池；完整官方 test 1,319 題用於最終評估。清洗、分詞與 loss mask 統一，候選評分僅使用候選池。

模型與資料集來源：

- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [GSM8K 原始實作與資料說明](https://github.com/openai/grade-school-math)

## 執行順序

1. 準備資料，固定模型及資料版本，核對長度與 mask 範例。
2. 評分並快取候選損失，生成 B/C 預算匹配的訓練清單。
3. 短試跑確認 GPU、顯存與步數計數。
4. B/C 各自從相同基礎模型訓練，保存預算末端 adapter。
5. 驗證三組生成格式及長度，固定共同設定。
6. 評估完整 test，重算指標、配對區間與案例，封存產物。

初始參數與實作規則見 [方法與評估](methodology.md)。每個階段保存設定、資料指紋、實際 token、時間與顯存。

## 結果呈現

交付包含方法流程圖、三組主結果表、成本比較圖與逐題案例。報告同時討論準確率、格式遵循和生成長度。單種子結果用作當前設定的觀察；擴充時補充訓練重複與超參數敏感度。

GSM8K 是公開基準，原模型的相關資料曝露情況未核實。後續可使用新的題目或改寫測試集檢查遷移表現。

課程原始交付要求保存在 [course-requirements.md](course-requirements.md)。
