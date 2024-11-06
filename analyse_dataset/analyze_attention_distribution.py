import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from utils import conv_to_text
from datasets import load_dataset
from tqdm.auto import tqdm, trange
import json
from jsonargparse import CLI
import warnings

def extract_attn_scores(outputs, layer_head_tuples: Tuple[int, int]=None):
    # aggregate over all layers and heads
    attn_scores = []
    if layer_head_tuples is None:
        for layer_attns in outputs.attentions:
            layer_attns = layer_attns.squeeze(0).detach().cpu().numpy() # [NUM_HEADS, LEN, LEN]
            attn_scores.append(layer_attns)
    else:
        for layer_id, head_id in layer_head_tuples:
            assert layer_id is not None
            if head_id is None:
                layer_attns = outputs.attentions[layer_id]
                layer_attns = layer_attns.squeeze(0).detach().cpu().numpy() # [NUM_HEADS, LEN, LEN]
                attn_scores.append(layer_attns)
            else:
                layer_attns = outputs.attentions[layer_id]
                layer_attns = layer_attns.squeeze(0)[head_id:head_id+1].detach().cpu().numpy() # [1, LEN, LEN]
                assert layer_attns.shape[0] == 1
                attn_scores.append(layer_attns)
    
    attn_scores = np.concatenate(attn_scores, axis=0) # [NUM_HEADS, LEN, LEN]
    
    return attn_scores

def check_attn_causal(attn):
    for i in range(len(attn)):
        if not np.isclose(attn[i, :i+1].sum(), 1):
            print(i, attn[i])
            raise ValueError(f"Causal attention should add up to 1 but got {attn[i, :i+1].sum()} instead.")
        if attn[i, i+1:].sum() != 0:
            attn[i+1:] = 0
            warnings.warn('Attention matrix is not causal. Manually setting upper triangle to 0.')

def conv_concat(conv, tokenizer, template='tinyllama'):
    messages = [msg['content'] for msg in conv['messages'] if msg['role'] in ['user', 'assistant']]
    if template == 'tinyllama':
        return f' '.join(messages) + f'{tokenizer.eos_token}'
    if template == 'llama-2':
        return f' '.join(messages) + f' {tokenizer.eos_token}'
    if template == 'llama-3':
        return f' '.join(messages) + f'{tokenizer.eos_token}'

def get_attr_distrs(conv, model, tokenizer, layer_head_tuples, sfted=True, avg_attn_over_response=True, add_system_prompt=True, template='tinyllama', verbose=False):

    if add_system_prompt and (conv["messages"][0]['role'] != 'system'):
        conv["messages"].insert(0, dict(role="system", content=""))
    conv['text'] = tokenizer.apply_chat_template(conv["messages"], tokenize=False)
    conv['raw_text'] = conv_concat(conv, tokenizer, template)
    # print(conv['text'])
    # print(conv['raw_text'])

    raw_tokens = [tokenizer.bos_token] + tokenizer.tokenize(conv['raw_text'])
    raw_inputs = tokenizer(conv['raw_text'], return_tensors='pt')

    tokens = [tokenizer.bos_token] + tokenizer.tokenize(conv['text'])
    inputs = tokenizer(conv['text'], return_tensors='pt')

    start_ptr = 0
    matched_substring_ids = []
    matched_substring_ids_ = []
    # print(raw_tokens)
    # print(tokens)

    # import pdb; pdb.set_trace()

    for i, (token_id, token) in enumerate(zip(raw_inputs['input_ids'][0].tolist()[::-1], raw_tokens[::-1])):
        while tokenizer.decode(inputs['input_ids'][0].tolist()[::-1][start_ptr]).strip(' ') != tokenizer.decode(token_id).strip(' '):
            if len(matched_substring_ids_) > 0:
                matched_substring_ids.append(matched_substring_ids_[::-1])
                matched_substring_ids_ = []
            start_ptr += 1
            if len(matched_substring_ids) == 2:
                break
        if len(matched_substring_ids) == 2:
                break

        matched_substring_ids_.append((token_id, len(inputs['input_ids'][0].tolist()) - 1 - start_ptr, 
                                       len(raw_inputs['input_ids'][0].tolist()) - 1 - i, token))
        start_ptr += 1
    # if len(matched_substring_ids_) > 0:
    #     matched_substring_ids.append(matched_substring_ids_[::-1])
    #     matched_substring_ids_ = []
    matched_substring_ids = matched_substring_ids[::-1]
    assert len(matched_substring_ids) == 2
    if verbose:
        for matched_substring_ids_ in matched_substring_ids:
            raw_ids = np.asarray([e[2] for e in matched_substring_ids_])
            _ids = np.asarray([e[1] for e in matched_substring_ids_])
            print("raw: ", ' '.join(np.asarray(raw_tokens)[raw_ids]))
            print("tmp: ", ' '.join(np.asarray(tokens)[_ids]))

    return_outputs = {}

    with torch.no_grad():
        outputs = model(**{k: v.to('cuda') for k, v in raw_inputs.items()}, output_attentions=True)
        attn = extract_attn_scores(outputs, layer_head_tuples=layer_head_tuples)

        del outputs
    
    rel_ids = [np.asarray([id_[2] for id_ in ids_]) for ids_ in matched_substring_ids]

    attn_distrs_no_template = []
    if avg_attn_over_response:
        target_ids = rel_ids[-1]
    else:
        # ['<0x0A>', 'The', '▁best', '▁thing', '▁to', '▁do', '▁in', '▁San', '▁Francisco', '▁is']
        # find "is" in the response prefix
        target_ids = []
        is_token = tokenizer.tokenize(" is")[-1] # '▁is' (TinyLlama, llama-2) or 'Ġis' (llama-3)
        for _id in rel_ids[-1]:
            if raw_tokens[_id] == is_token:
                target_ids.append(_id)
                break
        if len(target_ids) == 0:
            raise ValueError(raw_tokens, [raw_tokens[_id] for _id in rel_ids[-1]],
                             raw_tokens[target_ids[0]], tokenizer.tokenize(" is")[-1])

    for rel_id in target_ids:
        i = np.where(rel_ids[-1] == rel_id)[0][0]
        attn_ = attn[:,rel_id]
        
        user_ids = rel_ids[0]
        response_ids = rel_ids[1][:i + 1]
        bos_ids = np.asarray(list(np.arange(0,user_ids.min())))
        # bos_ids = [np.arange(0,user_ids.min())] # "<s>", "▁"


        if len(set(np.concatenate([bos_ids, user_ids, response_ids]))) != rel_id + 1:
            print(bos_ids)
            print(user_ids)
            print(response_ids)
            print((attn_ > 0).sum())
            print(rel_id)
            raise ValueError()
        retrieval_head_idx = attn_[:,user_ids].sum(-1).argmax()
        attn_distr_no_template = {'bos_token': attn_[retrieval_head_idx,bos_ids].sum(), 
                                  'user_prompt': attn_[retrieval_head_idx,user_ids].sum(), 
                                  'response_prompt': attn_[retrieval_head_idx,response_ids].sum(),}
        attn_ = None
        attn = None
        
        attn_distrs_no_template.append(attn_distr_no_template)
    avg_attn_distr_no_template = {k: np.mean([d[k] for d in attn_distrs_no_template]) for k in attn_distrs_no_template[0].keys()}
    return_outputs['without_template'] = avg_attn_distr_no_template
    
    if sfted:
        with torch.no_grad():
            outputs = model(**{k: v.to('cuda') for k, v in inputs.items()}, output_attentions=True)
            attn = extract_attn_scores(outputs, layer_head_tuples=layer_head_tuples)
            del outputs
        
        rel_ids = [np.asarray([id_[1] for id_ in ids_]) for ids_ in matched_substring_ids]
        
        attn_distrs, attn_distrs_rm_template = [], []
        if avg_attn_over_response:
            target_ids = rel_ids[-1]
        else:
            # ['<0x0A>', 'The', '▁best', '▁thing', '▁to', '▁do', '▁in', '▁San', '▁Francisco', '▁is']
            # find "is" in the response prefix
            target_ids = []
            is_token = tokenizer.tokenize(" is")[-1] # '▁is' (TinyLlama, llama-2) or 'Ġis' (llama-3)
            for _id in rel_ids[-1]:
                if tokens[_id] == is_token:
                    target_ids.append(_id)
                    break
            if len(target_ids) == 0:
                raise ValueError(tokens, [tokens[_id] for _id in rel_ids[-1]],
                                 tokens[target_ids[0]], tokenizer.tokenize(" is")[-1])
        
        for rel_id in target_ids:
            i = np.where(rel_ids[-1] == rel_id)[0][0]
            attn_ = attn[:,rel_id]
            
            bos_ids = [0] # "<s>"
            user_ids = rel_ids[0]
            response_ids = rel_ids[1][:i + 1]
            template_ids = np.asarray(list(np.arange(1, user_ids.min())) + list(np.arange(user_ids.max() + 1, response_ids.min())))

            if len(set(np.concatenate([bos_ids, user_ids, response_ids, template_ids]))) != rel_id + 1:
                print(bos_ids)
                print(user_ids)
                print(response_ids)
                print(template_ids)
                print((attn_ > 0).sum(), attn_.shape)
                print(rel_id)
                raise ValueError()

            attn_distr = {'bos_token': attn_[retrieval_head_idx,bos_ids].sum(), 
                          'user_prompt': attn_[retrieval_head_idx,user_ids].sum(), 
                          'response_prompt': attn_[retrieval_head_idx,response_ids].sum(),
                          'template_tokens': attn_[retrieval_head_idx,template_ids].sum()}
            attn_ = None
            attn = None

            Z = np.sum(list(attn_distr.values()))
            for k in attn_distr.keys():
                attn_distr[k] /= Z
            
            attn_distr_rm_template = {'bos_token': attn_distr['bos_token'].copy(), 
                                      'user_prompt': attn_distr['user_prompt'].copy(), 
                                      'response_prompt': attn_distr['response_prompt'].copy(),}
            Z = np.sum(list(attn_distr_rm_template.values()))
            for k in attn_distr_rm_template.keys():
                attn_distr_rm_template[k] /= Z

            attn_distrs.append(attn_distr)
            attn_distrs_rm_template.append(attn_distr_rm_template)
        avg_attn_distr = {k: np.mean([d[k] for d in attn_distrs]) for k in attn_distrs[0].keys()}
        avg_attn_distr_rm_template = {k: np.mean([d[k] for d in attn_distrs_rm_template]) for k in attn_distrs_rm_template[0].keys()}
        return_outputs['with_template'] = avg_attn_distr
        return_outputs['with_template_renormed'] = avg_attn_distr_rm_template
    
    gc.collect()
    torch.cuda.empty_cache()
    return return_outputs

def main(model_name: str, save_dir = "./attn_distr/", name_suffix: str=None,
         avg_attn_over_response: bool=False, suffix: str=None, 
         layer_id: int=None, head_id: int=None, template='tinyllama', force_overwrite: bool=False):
    
    
    if layer_id is None:
        layer_head_tuples = None
    else:
        layer_head_tuples = [(layer_id, head_id)]

    save_name = model_name.split("/")[-1]
    save_name += ("_avg-tokens" if avg_attn_over_response else "_single-token")
    if suffix is not None:
        suffix_name = '-'.join(suffix.rstrip(".").strip().split(' ')[-2:]) # in case suffix ends in a period
        save_name += f"_suffix-{suffix_name}"
    if name_suffix is not None:
        save_name += f"_{name_suffix}"
    
    save_json_path = os.path.join(save_dir, f"{save_name}.json")
    save_fig_path = os.path.join(save_dir, f"{save_name}.png")
    
    print(os.path.abspath(save_json_path))
    print(os.path.abspath(save_fig_path))

    sfted = ('sft' in model_name.lower() or 'lora' in model_name.lower() or 'chat' in model_name.lower() or 'instruct' in model_name.lower())
    if sfted:
        print(f"This is a finetuned chat model that uses chat templates.")

    if os.path.isfile(save_json_path) and (not force_overwrite):
        print(f"Loading previous results from {save_json_path}")
        with open(save_json_path, 'r') as f:
            avg_attn_distrs = json.load(f)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dset = load_dataset("ucla-cmllab/needle-in-a-haystack-4k")

        q_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", attn_implementation='eager', quantization_config=q_config)
        
        all_attn_distrs = None
        for i, conv in enumerate(tqdm(dset['train'])):
            if suffix is not None:
                conv['messages'][0]['content'] += suffix
            attn_distrs = get_attr_distrs(conv, model, tokenizer, sfted=sfted, 
                                          avg_attn_over_response=avg_attn_over_response,
                                          layer_head_tuples=layer_head_tuples,
                                          template=template, verbose=(i == 0))
            
            if all_attn_distrs is None:
                all_attn_distrs = {k: [] for k in attn_distrs.keys()}
            for k in attn_distrs.keys():
                all_attn_distrs[k].append(attn_distrs[k])
        avg_attn_distrs = {k: {k_: np.mean([v[k_] for v in vs]).astype(np.float64) for k_ in vs[0].keys()} 
                           for k, vs in all_attn_distrs.items()}
    
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, 'w') as f:
            json.dump(avg_attn_distrs, f, indent=4)

    labels = ['bos_token', 'user_prompt', 'response_prompt', 'template_tokens']
    titles = ['without_template', 'with_template_renormed', 'with_template',]
    titles = [title for title in titles if title in avg_attn_distrs]
    
    fig, axs = plt.subplots(1, len(avg_attn_distrs), figsize=(5*len(avg_attn_distrs), 4.5*1))
    
    for ax, title in zip(axs.flat if len(avg_attn_distrs) > 1 else [axs], titles):
        pie_data = avg_attn_distrs[title]
        labels_ = [l for l in labels if l in pie_data]
        ax.pie([pie_data[l] for l in labels_], labels=labels_, autopct='%1.1f%%')
        ax.set_title(title.replace("_", " "))
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, format='png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    CLI(main)

