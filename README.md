## Reference
https://github.com/HazyResearch/legalbench

https://github.com/sunnynexus/Search-o1
## Version
PyTorch  2.1.0
CUDA  12.1

```bash
conda env create -f environment.yml
```
```bash
python scripts/run_search_o1.py \
--dataset_name International \
--split test \
--max_search_limit 5 \
--max_turn 5 \
--top_k 10 \
--max_doc_len 3000 \
--use_jina True \
--model_path "Qwen/Qwen2.5-7B-Instruct" \
--jina_api_key  \
--serper_api_key "" \
--subset_num 100
```

```bash
python scripts/run_direct_gen.py \
--dataset_name International \
--split test \
--model_path "Qwen/Qwen2.5-7B-Instruct" \
--subset_num 100
```
