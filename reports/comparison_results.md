# Model Comparison Results

_QA eval sample: 50 questions._

## 1. Retriever comparison (QA mode)

| retriever | num_queries | recall@1 | recall@3 | recall@5 | mrr | avg_latency_s |
| --- | --- | --- | --- | --- | --- | --- |
| biomedclip | 50 | 0.0 | 0.06 | 0.14 | 0.046 | 0.3191 |

## 2. Report-generation strategy comparison

| strategy | bleu | rougeL |
| --- | --- | --- |
| zero_shot | 2.95 | 0.2512 |
| rag_fewshot | 6.66 | 0.2974 |
