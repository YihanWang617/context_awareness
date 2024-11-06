"""
Print the length of a task from huggingface
"""
from datasets import load_dataset
import argparse
from transformers import AutoTokenizer
import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--load_tokenizer_name', type=str, default='yihanwang617/tinyllama-sft-full-100k')
    parser.add_argument('--payload_key', type=str, default='messages')

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, download_mode='reuse_cache_if_exists')[args.split]
    tokenizer = AutoTokenizer.from_pretrained(args.load_tokenizer_name)
    messages = dataset[args.payload_key]


    bins = list(range(0, 2200, 100))
    cnt = [0 for b in bins]
    
    for message in messages:
        total_length = 0
        effective = False
        for item in message:
            if item['role'] in ['gpt', 'assistant'] and total_length <= 2048:
                effective = True
            total_length += len(tokenizer.encode(item['content']))
        if effective:
            cnt[min(total_length // 100, 21)] += 1
    
    print(bins)
    print(cnt)

    print(sum(cnt))

# ultrachat
# [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
# [10, 244, 955, 1949, 2867, 3769, 4568, 5479, 6211, 6645, 6739, 6514, 6243, 5859, 5514, 4921, 4531, 4142, 3531, 3202, 2716, 13263]

# vicuna
# [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
# [1894, 2296, 3125, 3605, 3277, 2908, 2713, 2736, 2495, 2381, 2192, 2025, 1956, 1968, 1910, 1759, 1817, 1827, 1763, 1991, 2025, 42233]
    

    