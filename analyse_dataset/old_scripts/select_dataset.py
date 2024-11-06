"""
Select a subset from a given dataset according to diversity/attention score.
"""
from task2vec import Task2Vec
from sklearn.cluster import KMeans
import random
import datasets
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, BitsAndBytesConfig
from utils import conv_to_text, tokenize, get_batch_user_attention, get_batch_length, build_completion_mask
from functools import partial
from task_similarity import get_normalized_embeddings
from tqdm import tqdm
import torch
import numpy as np
import pickle
import os
from datasets.utils.logging import disable_progress_bar
import math
disable_progress_bar()

def select_uniform(score_list, subset_size, num_bin=10):
    scores = [a[0] for a in score_list]
    max_score, min_score = int(max(scores)), int(min(scores))
    interval = (max_score - min_score)//num_bin + 1
    bin_split = range(min_score, max_score+1, interval)
    bins = [[] for i in bin_split]
    for s, b in score_list:
        bins[int(s - min_score)//interval].append((s, b))

    num_bin = len(bins)

    bin_cnt = [(len(bins[i]), i) for i in range(len(bins))]
    bin_cnt = sorted(bin_cnt)

    new_batch_list = []
    for i in range(num_bin):
        bin_index = bin_cnt[i][1]
        needed_subset_size = subset_size - len(new_batch_list)
        subsample_size = needed_subset_size//(num_bin - i)
        if subsample_size > len(bins[bin_index]):
            new_batch_list.extend(bins[bin_index])
        else:
            new_batch_list.extend(random.sample(bins[bin_index], min(subsample_size, len(bins[bin_index]))))
    
    return new_batch_list



def get_batch_length(batch, probe_model, probe_tokenizer):
    lengths = batch['attention_mask'].sum(dim=-1).to(torch.float)
    
    return lengths.mean().item()

def get_per_sample_user_length(examples, probe_model, probe_tokenizer, msg_key="messages", user_roles=["human", "user"], max_length=2048):
    avg_user_content_lengths = []
    for msgs in tqdm(examples[msg_key]):
        user_contents = [msg['content'] for msg in msgs if msg['role'] in user_roles]
        non_empty_user_contents = [context for content in user_contents if len(msg['content']) > 0]
        
        if len(non_empty_user_contents) == 0:
            avg_user_content_length = 0
        else:
            tokenized_contents = probe_tokenizer(non_empty_user_contents, return_tensors="pt", padding='max_length', 
                                                 truncation=True, max_length=max_length)
            # print(tokenized_user_contents['attention_mask'].shape, tokenized_user_contents['attention_mask'].sum(1))
            # for user_content, l in zip(user_contents, tokenized_user_contents['attention_mask'].sum(1)):
            #     print(l, user_content)
            avg_user_content_length = tokenized_contents['attention_mask'].sum().float() / len(user_contents)
        avg_user_content_lengths.append(avg_user_content_length)
    # examples['avg_user_content_length'] = avg_user_content_lengths
    return avg_user_content_lengths


def select_by_ranking(metric, tokenized_dataset, dataset, subset_size, probe_model, probe_tokenizer, **kwargs):
    batch_size = kwargs.get('batch_size', 512)
    order = kwargs.get('selection_order', 'large')
    select_with_raw_dataset = kwargs.get('select_with_raw_dataset', False)

    num_batches = len(dataset)//batch_size if select_with_raw_dataset else len(tokenized_dataset)//batch_size
    score_list = []

    metric_func = metric_dict[metric]

    if os.path.exists(f"cache/{dataset.info.dataset_name}_{metric}.pkl"):
        with open(f"cache/{dataset.info.dataset_name}_{metric}.pkl", 'rb') as fp:
            score_list = pickle.load(fp)
    else:
        for batch_idx in tqdm(range(num_batches)):
            if select_with_raw_dataset:
                batch = dataset.select(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
            else:
                batch = tokenized_dataset.select(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
            score = metric_func(batch, probe_model, probe_tokenizer)
            if score < 0:
                continue
            score_list.append((score, batch_idx))
        os.makedirs("./cache", exist_ok=True)
        with open(f"cache/{dataset.info.dataset_name}_{metric}.pkl", 'wb') as fp:
            pickle.dump(score_list, fp)


    if order in ['large']:
        score_list = [a for a in score_list if not math.isnan(a[0])]
        score_list = sorted(score_list, key=lambda x: x[0])
    if order in ['small']:
        score_list = [a for a in score_list if not math.isnan(a[0])]
        score_list = sorted(score_list, reverse=True, key=lambda x: x[0])
    if order == 'random':
        random.shuffle(score_list)

    new_dataset = []
    if order == 'large' or order == 'small' or order == 'random':
        while len(new_dataset) < subset_size and len(score_list) > 0:
            batch_idx = score_list[-1][1]
            new_batch = dataset.select(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
            for i in range(batch_size):
                new_dataset.append(new_batch[i])
            score_list = score_list[:-1]
    new_dataset = datasets.Dataset.from_list(new_dataset)
    return new_dataset

def select_by_per_sample_ranking(metric, tokenized_dataset, dataset, subset_size, probe_model, probe_tokenizer, **kwargs):
    # batch_size = kwargs.get('batch_size', 512)
    order = kwargs.get('selection_order', 'large')
    select_with_raw_dataset = kwargs.get('select_with_raw_dataset', False)

    # num_batches = len(dataset)//batch_size if select_with_raw_dataset else len(tokenized_dataset)//batch_size
    
    metric_func = metric_dict[metric]

    if os.path.exists(f"cache/{dataset.info.dataset_name}_{metric}.pkl"):
        with open(f"cache/{dataset.info.dataset_name}_{metric}.pkl", 'rb') as fp:
            score_list = pickle.load(fp)
    else:
        '''
        for batch_idx in tqdm(range(num_batches)):
            if select_with_raw_dataset:
                batch = dataset.select(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
            else:
                batch = tokenized_dataset.select(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
            scores = metric_func(batch, probe_model, probe_tokenizer)
            score_list += [(score, batch_idx * batch_size + i) for i, score in enumerate(scores)]
        '''
        if select_with_raw_dataset:
            scores = metric_func(dataset, probe_model, probe_tokenizer)
        else:
            scores = metric_func(tokenized_dataset, probe_model, probe_tokenizer)
        score_list = [(score, i) for i, score in enumerate(scores)]
        os.makedirs("./cache", exist_ok=True)
        with open(f"cache/{dataset.info.dataset_name}_{metric}.pkl", 'wb') as fp:
            pickle.dump(score_list, fp)

    if order in ['large']:
        score_list = [a for a in score_list if not math.isnan(a[0])]
        score_list = sorted(score_list)
    if order in ['small']:
        score_list = [a for a in score_list if not math.isnan(a[0])]
        score_list = sorted(score_list, reverse=True)
    if order == 'random':
        random.shuffle(score_list)

    new_dataset = []
    if order == 'large' or order == 'small' or order == 'random':
        new_dataset = dataset.select([idx for score, idx in score_list[-subset_size:]])
    else:
        raise NotImplementedError
    return new_dataset

def select_by_battle(metric, tokenized_dataset, dataset, subset_size, probe_model, probe_tokenizer, **kwargs):
    batch_size = kwargs.get('batch_size', 512)
    order = kwargs.get('selection_order', 'large')

    num_batches = len(tokenized_dataset)//batch_size

    metric_func = metric_dict[metric]

    new_dataset = []
    available_batch_idx = set(range(num_batches))

    for i in tqdm(range(subset_size)):
        batch_idx = random.sample(available_batch_idx, 2)
        batch = tokenized_dataset.select(batch_idx)
        new_batch = dataset.select(batch_idx)

        if metric == 'attention':
            max_length = batch['attention_mask'].sum(-1).min().item()
            batch = batch.map(lambda x: {'attention_mask': x['attention_mask'][:max_length], 'input_ids': x['input_ids'][:max_length]})

        score_1 = metric_func(batch[0:1], probe_model, probe_tokenizer)
        score_2 = metric_func(batch[1:2], probe_model, probe_tokenizer)
        
        if score_1 > score_2:
            if order == 'large':
                new_dataset.append(new_batch[0])
                available_batch_idx.discard(batch_idx[0])
            else:
                new_dataset.append(new_batch[1])
                available_batch_idx.discard(batch_idx[1])
        elif score_1 < score_2:
            if order == 'large':
                new_dataset.append(new_batch[1])
                available_batch_idx.discard(batch_idx[1])
            else:
                new_dataset.append(new_batch[0])
                available_batch_idx.discard(batch_idx[0])
        else:
            if random.random() < 0.5:
                new_dataset.append(new_batch[0])
                available_batch_idx.discard(batch_idx[0])
            else:
                new_dataset.append(new_batch[1])
                available_batch_idx.discard(batch_idx[1])


    new_dataset = datasets.Dataset.from_list(new_dataset)
    return new_dataset
    
metric_dict = {
    'length': get_batch_length,
    'user_attention': get_batch_user_attention,
    'per_sample_user_length': get_per_sample_user_length,
}

selection_dict = {
    'ranking': select_by_ranking,
    'per_sample_ranking': select_by_per_sample_ranking,
    'battle': select_by_battle,
}

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ucla-cmllab/vicuna_cleaned')
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--num_sample_from', type=int, default=5)
    parser.add_argument('--subset_size', type=int, default=50000)
    parser.add_argument('--push_to_hub', type=str)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--metric', type=str, choices=['length', 'user_attention', 'per_sample_user_length'])
    parser.add_argument('--selection_mode', type=str, choices=['ranking', 'battle', 'per_sample_ranking'])
    parser.add_argument('--selection_order', type=str, choices=['large', 'small', 'random'])
    parser.add_argument('--probe_model', type=str, default='gpt2')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--ori_set_size', type=int, default=None)
    args = parser.parse_args()

    if args.metric in ['length', 'user_length']:
        probe_network = None
    else:
        if args.quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            probe_network = AutoModelForCausalLM.from_pretrained(args.probe_model, quantization_config=quantization_config, device_map='auto')
        else:
            probe_network = AutoModelForCausalLM.from_pretrained(args.probe_model).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.probe_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, split="train", download_mode='reuse_cache_if_exists')
    test_dataset = load_dataset(args.dataset_name, split="test_sft", download_mode='reuse_cache_if_exists')

    # make sure that every training example starts from a user input.
    dataset = dataset.map(lambda example: example if len(example['messages']) == 0 or example['messages'][0]['role'] in ['user', 'human', 'system'] else {'id': example['id'], 'messages': example['messages'][1:]})

    dataset = dataset.map(partial(conv_to_text, template='TinyLlama'))
    dataset = dataset.map(partial(tokenize, tokenizer=tokenizer, max_length=args.max_length), batched=True)

    dataset = dataset.shuffle(42)

    if args.ori_set_size is not None:
        dataset = dataset.select(list(range(args.ori_set_size)))

    tokenized_dataset = dataset.select_columns(['input_ids', 'attention_mask'])
    tokenized_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])

    dataset = dataset.remove_columns(['input_ids', 'attention_mask', 'text'])

    select_with_raw_dataset = args.metric in ['user_length']
    
    select_function = selection_dict[args.selection_mode]
    sampled_dataset = select_function(args.metric, tokenized_dataset, dataset, args.subset_size, probe_network, tokenizer, num_clusters=args.num_clusters, num_sample_from = args.num_sample_from, batch_size=args.batch_size, selection_order=args.selection_order, select_with_raw_dataset=select_with_raw_dataset)

    sampled_dataset = DatasetDict({'train': sampled_dataset, 'test': test_dataset})
    sampled_dataset.push_to_hub(args.push_to_hub, private=True)

if __name__ == "__main__":
    main()


