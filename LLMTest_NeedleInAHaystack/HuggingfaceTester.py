import os
import torch
from LLMNeedleHaystackTester import LLMNeedleHaystackTester
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.conversation import get_conv_template
from utils import conv_template_dict, smart_tokenizer_and_embedding_resize
from argparse import ArgumentParser
from needle_config import needle_dict

import sys
sys.path.append("../PASTA")

class HuggingfaceTester(LLMNeedleHaystackTester):
    def __init__(self, **kwargs):
        if (("evaluation_method" not in kwargs) or (kwargs["evaluation_method"] == "gpt4")) and \
           ("openai_api_key" not in kwargs and not os.getenv("OPENAI_API_KEY")):
            raise ValueError(
                "Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env"
            )
            self.openai_api_key = kwargs.pop("openai_api_key", os.getenv("OPENAI_API_KEY"))

        if "model_name" not in kwargs:
            raise ValueError(
                "model_name must be supplied with init"
            )

        if "evaluation_method" not in kwargs:
            print(
                "since evaluation method is not specified , 'gpt4' will be used for evaluation"
            )
        elif kwargs["evaluation_method"] not in ("gpt4", "substring_match"):
            raise ValueError("evaluation_method must be 'substring_match' or 'gpt4'")

        self.model_name = kwargs["model_name"]
        self.model_to_test_description = kwargs.pop("model_name")
        
        model_load_kwargs = {}
        if 'attn_implementation' in kwargs:
            model_load_kwargs['attn_implementation'] = kwargs.pop('attn_implementation')

        for model_name in ['llama-2-70b', 'llama-2', 'llama-3', 'gemma', 'gemma-2', 'qwen2.5']:
            if model_name in self.model_name.lower():
                model_load_kwargs['torch_dtype'] = torch.bfloat16

        self.model_to_test = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', **model_load_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        if 'llama-3' in self.model_name.lower():
            from pastalib.utils.CustomLlama3Tokenizer import CustomLlama3Tokenizer
            self.tokenizer = CustomLlama3Tokenizer(self.tokenizer)


        
        self.template = kwargs.pop("template")
        self.add_hint = kwargs.pop('add_hint')
        if 'gemma' in self.model_name.lower():
            print("Gemma-7b-t does not support system prompts.")
            kwargs.pop('add_system_prompt')
            self.add_system_prompt = False 
        else:
            self.add_system_prompt = kwargs.pop('add_system_prompt', True)

        super().__init__(**kwargs)

    def get_encoding(self, context):
        return self.tokenizer.encode(context)[1:]

    def get_decoding(self, encoded_context):
        return self.tokenizer.decode(encoded_context)

    def get_prompt(self, context, return_context=False):
        conv = []

        if self.add_system_prompt:
            conv.append(dict(role='system', content=''))
        content = f"You are a helpful AI assistant that answers a question using only the provided document: \n{context}\n\nQuestion: {self.retrieval_question}"
        conv.append(dict(role='user', content=content))

        if self.add_hint:
            hint = " ".join(self.needle.split(" ")[:8])
            if self.template == 'raw':
                for message in conv:
                    if message['role'] == 'user':
                        user_prompt = message['content']
                prompt = user_prompt + "\n\n" + hint
            else:
                conv.append(dict(role='assistant', content=hint))
                prompt = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                prompt = prompt.strip().removesuffix(f'{self.tokenizer.bos_token}').removesuffix(f'{self.tokenizer.eos_token}').removesuffix(f'{self.tokenizer.bos_token}').removesuffix('<end_of_turn>').strip()
        else:
            if self.template == 'raw':
                prompt = user_prompt
            else:
                prompt = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        if return_context:
            return prompt, content
        return prompt

    async def get_response_from_model(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        generate_ids = self.model_to_test.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.pad_token_id)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        '''
        conv = get_conv_template(self.template)
        # vicuna_v1.1 llama-2 zero_shot TinyLlama zephyr raw
        
        stop_str = conv.stop_str
        if stop_str:
            if isinstance(stop_str, list):
                for stop in stop_str:
                    pos = response.find(stop)
                    if pos != -1:
                        response = response[:pos]
            elif isinstance(stop_str, str):
                pos = response.find(stop_str)
                if pos != -1:
                    response = response[:pos]
            else:
                raise ValueError()
        '''
        return response

def main(model_name, template, needle_name, evaluation_method="substring_match", context_lengths_min=200, context_lengths_max=4000, add_hint=False, add_system_prompt=True, save_model_suffix=None):
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = HuggingfaceTester(model_name=model_name, 
                           template=template,
                           evaluation_method=evaluation_method, 
                           context_lengths_min=context_lengths_min, 
                           context_lengths_max=context_lengths_max,
                           start_context_lengths=context_lengths_min,
                           needle_name=needle_name,
                           add_hint=add_hint,
                           add_system_prompt=add_system_prompt,
                           save_model_suffix=save_model_suffix)
    ht.start_test()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', help='path of the model on huggingface hub')
    parser.add_argument('--template', choices=["raw", "default"], default="default")
    parser.add_argument('--needle_name', default="SF")
    parser.add_argument('--add_hint', action='store_true')
    parser.add_argument('--context_lengths_max', type=int, default=4000)
    parser.add_argument('--save_model_suffix', type=str, default=None)
   
    args = parser.parse_args()

    main(model_name=args.model_name, template=args.template, needle_name=args.needle_name, add_hint=args.add_hint, context_lengths_max=args.context_lengths_max, save_model_suffix=args.save_model_suffix)

