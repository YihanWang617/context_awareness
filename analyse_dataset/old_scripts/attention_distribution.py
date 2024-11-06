from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastchat.conversation import get_conv_template
from datasets import load_dataset
from argparse import ArgumentParser
import torch
from matplotlib import pyplot as plt
import statistics
from tqdm.auto import tqdm

def get_top_attention(model, tokenizer, input_text, layer_index=10, head_index=None, top_k=1):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.encode(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=2048)  # Tokenize input text
    outputs = model(inputs, output_attentions=True)  # Run model

    if head_index == None:
        attention = outputs[-1][layer_index][0]
    else:
        attention = outputs[-1][layer_index][0][head_index:head_index+1]  # Retrieve attention from model outputs

    attention_mask = (inputs == 13) | (inputs == 1) | (inputs == 2)

    attention[:,:,attention_mask[0]] = 0.0
    if inputs.shape[1] < top_k:
        return inputs, inputs
    top_attention = attention.topk(dim=-1, k=top_k)[1]
    return top_attention, inputs

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_stats_npz', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default="train")
    
    parser.add_argument('--model_name', type=str, default='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    parser.add_argument('--conv_template_name', type=str, default="TinyLlama")
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--seq_length', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1126)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quantization_config, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    '''
    ratio_dict = {}
    length_dict = {}
    for dataset_name in args.dataset_name:
    '''
    
    dataset = load_dataset(args.dataset_name)[args.split]
    dataset = dataset.shuffle(seed=args.seed).select(range(args.batch_size))

    ratios = [[] for i in range(args.seq_length)]
    lengths = []

    for item in tqdm(dataset):
        conv = get_conv_template(args.conv_template_name)
        for m in item['messages']:
            if m['role'] in ['human', 'user']:
                conv.append_message(conv.roles[0], m['content'])
            else:
                conv.append_message(conv.roles[1], m['content'])
        prompt = conv.get_prompt()

        top_attention, inputs = get_top_attention(model, tokenizer, prompt, top_k=args.top_k)
        length = inputs.shape[1]

        if length <= args.seq_length: 
            continue

        gap = torch.Tensor(list(range(top_attention.shape[1]))).unsqueeze(1) - top_attention
        max_gap = gap.max(-1)[0] # max gap in top_k
        max_gap = max_gap[:, args.seq_length:2 * args.seq_length].median(dim=0)[0] # median for H heads


        for i, token_max_gap in enumerate(max_gap):
            ratios[i].append(token_max_gap.item())
        lengths.append(length)

    ratios = np.asarray([statistics.median(a) for a in ratios]) # median for N examples
    lengths = np.asarray(lengths)

    os.makedirs(os.path.dirname(args.save_stats_npz), exist_ok=True)
    np.savez_compressed(args.save_stats_npz, ratios=ratios, lengths=lengths)
    
    '''
    ratio_dict[dataset_name] = ratios
    length_dict[dataset_name] = lengths

    cnt = 0
    for i in range(1000):
        if ratio_dict[args.dataset_name[0]][i] >= ratio_dict[args.dataset_name[1]][i]:
            cnt += 1
    print(args.dataset_name)
    print(f"{cnt}/1000")

    cnt = 0
    for i in range(1000):
        if ratio_dict[args.dataset_name[1]][i] >= ratio_dict[args.dataset_name[0]][i]:
            cnt += 1

    print(f"{cnt}/1000")

    import pdb; pdb.set_trace()
    print(length_dict)
    1: 979/1000, 10: 943/1000
    '''
    
if __name__ == '__main__':
    main()
