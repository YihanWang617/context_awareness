from HuggingfaceTester import HuggingfaceTester
from fastchat.conversation import get_conv_template
import numpy as np
import torch
import os
from typing import List
from jsonargparse import CLI
import asyncio
from pastalib.pasta import PASTA 

class MyHuggingfaceTester(HuggingfaceTester):
    def generate(self, prompt, max_new_tokens=60, alpha=0.01):
        head_config = {
            "15": [3, 18, 19, 20, 21], 
            "18": [1, 4, 7, 30], 
        }

        # Initialize the PASTA steerer
        pasta = PASTA(
            model=self.model_to_test,
            tokenizer=self.tokenizer,
            head_config=head_config, 
            alpha=alpha, # 0.01, scaling coefficient
            scale_position="exclude",
        )
        inputs, offset_mapping = pasta.inputs_from_batch(prompt)
        inputs = inputs.to("cuda")
        # User highlights specific input spans
        emphasized_texts = ["eat a sandwich and sit in Dolores Park on a sunny day"]
        # PASTA registers the pre_forward_hook to edit attention
        with pasta.apply_steering(
            model=self.model_to_test, 
            strings=prompt, 
            substrings=emphasized_texts, 
            model_input=inputs,
            offsets_mapping=offset_mapping
        ) as steered_model: 
            outputs = steered_model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_attentions=True)
        # outputs.attentions: Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor of shape (batch_size, num_heads, generated_length, sequence_length)
        # print(outputs.attentions[-1][15][0, 19, -1, :])
        output = outputs.sequences[0][inputs.input_ids.shape[1]:]

        '''
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        input_ids = inputs.input_ids
        with torch.no_grad():
            q_outputs = self.model_to_test(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)
            inp = input_ids[:,-1]
            output = []
            past_kv = q_outputs.past_key_values
            for step_i in range(max_new_tokens):
                inp = inp.view(1, 1)
                outputs = self.model_to_test(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True)
                past_kv = outputs.past_key_values
                inp = outputs.logits[0, -1].argmax()
                step_token = self.tokenizer.convert_ids_to_tokens(inp.item())
                output.append(inp.item())
                # outputs.attentions: [n_layers] + [batch_size, num_heads, sequence_length, sequence_length]
                # see more details in https://github.com/huggingface/transformers/blob/e65502951593a76844e872fee9c56b805598538a/src/transformers/modeling_outputs.py#L698
                if step_token=='<0x0A>' or inp.item()==144: break
        '''
        return output

    async def get_response_from_model(self, prompt):
        # generate_ids = self.model_to_test.generate(inputs.input_ids, max_new_tokens=60, pad_token_id=self.tokenizer.pad_token_id)
        # generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        generate_ids = self.generate(prompt, max_new_tokens=60)
        response = self.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

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
        return response

def main(model_name: str=None, template: str=None, needle_name: str="SF", evaluation_method: str="substring_match", 
         context_lengths: List[int]=[200] * 18, 
         depth_percents: List[int]=[41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 79, 82, 85, 88, 91, 94, 97, 100], add_hint: bool=False,):

    ht = MyHuggingfaceTester(
        model_name=model_name, template=template,
        evaluation_method=evaluation_method, 
        context_lengths=context_lengths, # [context_length], 
        document_depth_percents=depth_percents, # [depth_percent],
        needle_name=needle_name,
        add_hint=add_hint)

    for context_length, depth_percent in zip(context_lengths, depth_percents):
        context = asyncio.run(ht.generate_context(context_length, depth_percent))
        prompt = ht.get_prompt(context)
        response = asyncio.run(ht.get_response_from_model(prompt))
        print(response)

if __name__ == '__main__':
    CLI(main)

