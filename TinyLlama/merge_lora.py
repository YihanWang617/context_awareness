from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import get_peft_model
import sys
sys.path.insert(0, './')
from scripts.utils import smart_tokenizer_and_embedding_resize
import argparse
from peft import PeftModel
import sys
sys.path.append('./')


def merge_model(base_model_path, lora_path, commit, push_hub):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    if  tokenizer.eos_token_id == tokenizer.pad_token_id or tokenizer.pad_token_id is None:

        # add pad tokens
        special_tokens_dict = dict(pad_token='[PAD]')
        non_sepcial_tokens_dict = []
        if 'indicator' in lora_path:
            non_sepcial_tokens_dict = ['[IND]']

        smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                non_special_tokens=non_sepcial_tokens_dict,
                tokenizer=tokenizer,
                model=base_model
            )
            

    merged_model = PeftModel.from_pretrained(base_model, lora_path, revision=commit)
    merged_model = merged_model.merge_and_unload()
    import pdb; pdb.set_trace()
    merged_model.push_to_hub(push_hub, use_temp_dir=False)
    tokenizer.push_to_hub(push_hub)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--commit', type=str, default='main')
    parser.add_argument('--push_hub', type=str, default=None)

    args = parser.parse_args()

    if args.push_hub is None:
        args.push_hub = args.lora_path

    merge_model(args.base_model, args.lora_path, args.commit, args.push_hub)