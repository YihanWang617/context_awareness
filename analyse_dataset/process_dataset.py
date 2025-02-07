from datasets import load_dataset, Dataset, DatasetDict, get_dataset_split_names
from tqdm import tqdm
from argparse import ArgumentParser
import datasets
from utils import conv_to_text, get_batch_user_attention, tokenize
import json
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
datasets.disable_caching()

def vicuna_to_ultrachat(item, **kwargs):
    new_item = {}
    if 'conversations' in item:
        conv = item.pop('conversations')
    elif 'messages' in item:
        conv = item.pop('messages')
    
    new_conv = []
    for conv_msg in conv:
        new_conv_msg = {}
        if 'from' in conv_msg:
            conv_msg['role'] = conv_msg['from']
        if 'value' in conv_msg:
            conv_msg['content'] = conv_msg['value']
        new_conv_msg['role'] = 'user' if conv_msg['role'] in ['human', 'user'] else 'assistant'
        new_conv_msg['content'] = conv_msg['content']
        new_conv.append(new_conv_msg)

    new_item = {'messages': new_conv}
    new_item.update(item)

    return new_item


def wizardLM_to_ultrachat(item, **kwargs):
    new_conv = [{'role': 'user', 'content': item['instruction']}, {'role': 'assistant', 'content': item['output']}]
    new_item = {'messages': new_conv}

    return new_item


def add_prompt(item, model, tokenizer, max_token_length=2047, layer_idx=15):
    new_item = {}
    prompt_text = conv_to_text(item, tokenizer)
    conv = item.pop('messages')
    batch = tokenizer([prompt_text], return_tensors="pt", padding='max_length', truncation=True, max_length=max_token_length)
    avg_user_attention, user_attentions = get_batch_user_attention(batch, model, tokenizer, return_separate=True, layer_index=layer_idx)
    for idx, conv_msg in enumerate(conv):
        role = conv_msg['role']
        if role == 'assistant':
            if len(user_attentions) == 0:
                break
            additional_prompt = f"The user attention ratio is {user_attentions[0]}."
            #Base your answer {int(user_attentions[0]*100)}% on the given information and {100-int(user_attentions[0]*100)}% on your own knowledge."
            if idx - 1 >= 0:
                conv[idx-1]['content'] = conv[idx-1]['content'] + " " + additional_prompt
            user_attentions = user_attentions[1:]
    new_item = {'messages': conv}
    new_item.update(item)
    if 'text' in new_item:
        new_item.pop('text')

    return new_item


def remove_prompt(item, max_token_length=2047):
    conv = item['messages']
    
    for i, conv_msg in enumerate(conv):
        conv_msg['content'] = conv_msg['content'].removesuffix(' Base your answer more on the given information.')
        conv_msg['content'] = conv_msg['content'].removesuffix(' Base your answer more on your own knowledge.')

    return item

positive_cnt=negative_cnt=none_cnt=context_cnt=0

def modify_prompt(item, threshold=0.5, max_token_length=2047):
    global positive_cnt, negative_cnt, none_cnt, context_cnt
    new_item = {}
    conv = item.pop('messages')
    
    for i, conv_msg in enumerate(conv):
        role = conv_msg['role']
        if role == 'user':
            if "The user attention ratio is " in conv_msg['content']:
                conv_msg['content'], prompt = " ".join(conv_msg['content'].split(' ')[:-6]), conv_msg['content'].split(' ')[-6:]
                ratio = float(prompt[-1][:-1])
                if ratio > threshold:
                    new_prompt = "[IND]" #f"Base your answer more on the given information."
                    positive_cnt += 1
                else:
                    new_prompt = ""
                    negative_cnt += 1
                if len(conv_msg['content']) > 0 and conv_msg['content'][-1].isalnum():
                    conv_msg['content'] += '.'
                conv_msg['content'] += " " + new_prompt if new_prompt != "" else ""
    new_item = {'messages': conv}
    new_item.update(item)
    if 'text' in new_item:
        new_item.pop('text')

    return new_item


preprocess_dict = {
    'None': lambda x, **kwargs: x,
    'vicuna_to_ultrachat': vicuna_to_ultrachat,
    'add_prompt': add_prompt,
    'modify_prompt': modify_prompt,
    'remove_prompt': remove_prompt,
    'wizardLM_to_ultrachat': wizardLM_to_ultrachat
}

def main(dataset_name, preprocess_function_name, split_to_process, train_size, save_huggingface_hub, configs = "{}", keep_original_splits = ["train", 'test'], subset_size=100000, streaming=False, max_token_length=2047, layer_idx=15):
    original_splits = get_dataset_split_names(dataset_name)
    new_dataset = {}
    for split in original_splits:
        if split != split_to_process:
            if split in keep_original_splits:
                new_dataset[split] = load_dataset(dataset_name, split=split)
                new_dataset[split] = new_dataset[split].select(range(min(subset_size, len(new_dataset[split]))))
            continue

        
        dataset = load_dataset(dataset_name, split=split_to_process, streaming=streaming)
        # make sure that every training example starts from a user input.
        # dataset = dataset.map(lambda example: example if len(example['messages']) == 0 or example['messages'][0]['role'] in ['user', 'human'] else {'id': example['id'] if 'id' in example else 0, 'messages': example['messages'][1:]})

        dataset = dataset.shuffle(seed=42)

        sampled_dataset = []

        if not streaming:
            sampled_dataset = dataset.select(range(min(subset_size, len(dataset))))
        else:
            for i in tqdm(range(subset_size)):
                item = next(iter(dataset))
                sampled_dataset.append(item)
        
            sampled_dataset = Dataset.from_list(sampled_dataset)
        
        preprocess_config = json.loads(configs)
        if 'probe_model' in preprocess_config:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(preprocess_config['probe_model'], quantization_config=quantization_config, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(preprocess_config['probe_model'])
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            preprocess_func = partial(preprocess_dict[preprocess_function_name], model=model, tokenizer=tokenizer, max_token_length=max_token_length, layer_idx=layer_idx)
        else:
            preprocess_func = partial(preprocess_dict[preprocess_function_name], max_token_length=max_token_length, **preprocess_config)
        
        sampled_dataset = sampled_dataset.map(preprocess_func)
        if len(sampled_dataset) - train_size > 0:
            sampled_dataset = sampled_dataset.train_test_split(test_size=len(sampled_dataset) - train_size)
            new_dataset['train'] = sampled_dataset['train']
            new_dataset['test'] = sampled_dataset['test']
        else:
            new_dataset['train'] = sampled_dataset

    if 'test' in new_dataset:
        new_dataset['test'] = new_dataset['test'].map(preprocess_func)
    if 'test_sft' in new_dataset:
        new_dataset['test_sft'] = new_dataset['test_sft'].map(preprocess_func)

    if 'text' in new_dataset['train'].column_names:
        new_dataset['train'] = new_dataset['train'].remove_columns(['text'])
    if 'text' in new_dataset['test'].column_names:
        new_dataset['test'] = new_dataset['test'].remove_columns(['text'])
    new_dataset = DatasetDict(new_dataset)
    import pdb; pdb.set_trace()
    new_dataset.push_to_hub(save_huggingface_hub, private=True)

    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cerebras/SlimPajama-627B')
    parser.add_argument('--preprocess_function_name', type=str, default='None')
    parser.add_argument('--split_to_process', type=str, default='train')
    parser.add_argument('--save_huggingface_hub', type=str)
    parser.add_argument('--train_size', type=int, default=500000)
    parser.add_argument('--subset_size', type=int, default=500000)
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--configs', type=str, default="{}")
    parser.add_argument('--max_token_length', type=int, default=2047)
    parser.add_argument('--layer_idx', type=int, default=15)

    args = parser.parse_args()
    main(args.dataset_name, args.preprocess_function_name, args.split_to_process, args.train_size, args.save_huggingface_hub, configs=args.configs, subset_size = args.subset_size, streaming=args.streaming, max_token_length=args.max_token_length, layer_idx=args.layer_idx)

# python process_dataset.py --dataset_name cerebras/SlimPajama-627B