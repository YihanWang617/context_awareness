"""
Select a subset from a given dataset according to diversity/attention score.
"""
from task2vec import Task2Vec
from argparse import ArgumentParser
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import conv_to_text, tokenize, get_batch_user_attention
from functools import partial
from tqdm import tqdm
from statistics import mean
import math



def analyze_user_attention(tokenized_dataset, probe_model, **kwargs):
    probe_tokenizer = kwargs.get('probe_tokenizer', None)
    layer_index = kwargs.get('layer_index', 15)
    batch_size = 1

    num_batches = len(tokenized_dataset)//1
    score_list = []
    for batch_idx in tqdm(range(num_batches)):
        batch = tokenized_dataset.select(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        score, scores = get_batch_user_attention(batch, probe_model, probe_tokenizer, return_separate=True, layer_index=layer_index)
        if not math.isnan(scores[0]):
            score_list.append(scores[0])
    
    print(f"average attention distance: {mean(score_list)}")


def analyze_length(tokenized_dataset, probe_model, **kwargs):
    bins = list(range(0, kwargs['max_length'], 100))
    cnt = {b: 0 for b in bins}

    lengths = tokenized_dataset['attention_mask'].sum(dim=-1)
    lengths = [l.item() for l in lengths]
    
    total = 0
    for l in lengths:
        cnt[min(l // 100, kwargs['max_length']//100) * 100] += 1
        total += l

    print("count of examples in different length bins: ", cnt)
    print("average length in tokens: ", total/len(lengths))
    print("median length in tokens: ", total/len(lengths))
    print("total examples: ", len(lengths))


analyze_dict = {
    'user_attention': analyze_user_attention,
    'length': analyze_length,
}
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='yihanwang617/vicuna_cleaned')
    parser.add_argument('--revision', type=str, default="main")
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--metric', type=str, choices=['diversity', 'user_attention', 'length', 'rope', 'perplexity'])
    parser.add_argument('--probe_model', type=str, default='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--subset_size', type=int, default=1000000)
    parser.add_argument('--head_index', type=int, default=19)
    parser.add_argument('--layer_index', type=int, default=15)

    args = parser.parse_args()

    if args.quantize:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        probe_network = AutoModelForCausalLM.from_pretrained(args.probe_model, quantization_config=quantization_config)
    else:
        probe_network = AutoModelForCausalLM.from_pretrained(args.probe_model).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.probe_model)

    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, split="train", revision=args.revision, download_mode='reuse_cache_if_exists')

    dataset = dataset.shuffle(42)

    dataset = dataset.select(range(min(args.subset_size, len(dataset))))
    
    # make sure that every training example starts from a user input.
    dataset = dataset.map(lambda example: example if len(example['messages']) == 0 or example['messages'][0]['role'] in ['user', 'human'] else {'id': example['id'] if 'id' in example else 0, 'messages': example['messages'][1:]})

    # import pdb; pdb.set_trace()
    removed_columns = dataset.column_names
    removed_columns.remove('messages')
    dataset = dataset.map(lambda x: {'messages': [[x['messages'][i][j]] for i in range(len(x['messages'])) for j in range(len(x['messages'][i])//2)]}, batched=True, remove_columns=removed_columns)
    # import pdb; pdb.set_trace()
    # dataset = dataset.map(lambda x: {'messages': [(x['messages'][i][j], x['messages'][i][j+1]) for i in range(len(x['messages'])) for j in range(len(x['messages'][i])//2)]}, batched=True, remove_columns=removed_columns)
    # dataset = dataset.filter(lambda x: '[IND]' in x['messages'][0]['content'])
    dataset = dataset.map(partial(conv_to_text, probe_tokenizer=tokenizer))
    dataset = dataset.map(partial(tokenize, tokenizer=tokenizer, max_length=args.max_length), batched=True)


    tokenized_dataset = dataset.select_columns(['input_ids', 'attention_mask'])
    tokenized_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])

    dataset = dataset.remove_columns(['input_ids', 'attention_mask', 'text'])

    analyze_function = analyze_dict[args.metric]
    analyze_function(tokenized_dataset, probe_network, probe_tokenizer=tokenizer, batch_size=args.batch_size, layer_index=args.layer_index, head_index=args.head_index, max_length=args.max_length)


if __name__ == "__main__":
    main()


