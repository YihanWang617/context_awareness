from typing import Optional
from HuggingfaceTester import HuggingfaceTester
import numpy as np
import torch
import os
from typing import List
from jsonargparse import CLI
import asyncio
from pastalib.pasta import PASTA
import sys
sys.path.append('../')
from UserSteeredWrapper import UserSteeredWrapper

class PastaTester(HuggingfaceTester):
    PASTA_HEAD_CONFIGS = {
        "tinyllama": {k: None for k in range(22)},
        "llama-2-7b": {k: None for k in range(32)},
        "llama-2-13b": {k: None for k in range(32)},
        "llama-3": {k: None for k in range(32)},
    }
    
    @staticmethod
    def get_pasta_head_config(model_name):
        for k, v in PastaTester.PASTA_HEAD_CONFIGS.items():
            if k in model_name.lower():
                return v
        return None

    def __init__(self, **kwargs):
        self.alpha = kwargs.pop("alpha")
        assert (self.alpha <= 1) and (self.alpha > 0)
        super().__init__(**kwargs)
        head_config = self.get_pasta_head_config(self.model_name)
        if head_config is None:
            raise ValueError(f"Pasta head config not found for {self.model_name}. Populate PASTA_HEAD_CONFIGS with retrieval heads for this model before running HayStack with Pasta intervention.")
        self.wrapped_model = UserSteeredWrapper(head_config, self.model_to_test, self.tokenizer, self.alpha)

    def generate(self, conversation_history, context=None, max_new_tokens=100):
        # print(prompt)
        
        output = self.wrapped_model.generate(conversation_history, context, max_new_tokens=max_new_tokens)
        
        return output

    async def get_response_from_model(self, prompt_context):
        # generate_ids = self.model_to_test.generate(inputs.input_ids, max_new_tokens=60, pad_token_id=self.tokenizer.pad_token_id)
        # generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        prompt, context = prompt_context
        generate_ids = self.generate(prompt, context, max_new_tokens=100)
        response = self.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return response

    def get_prompt(self, context, return_context=True):
        prompt, context = super().get_prompt(context, return_context=True)
        if return_context:
            return prompt, context
        else:
            return prompt

def main(model_name: str=None, template: str='default', needle_name: str="SF", evaluation_method: str="substring_match", 
         alpha: float=0.01, context_lengths_min: int=200, context_lengths_max: int=4000, start_context_lengths=200,
         context_lengths: Optional[List[int]]=None, document_depths: Optional[List[int]]=None,
         add_hint: bool=False, save_model_suffix: Optional[str]=None):

    assert model_name is not None
    assert template in ["raw", "default"]

    if save_model_suffix is None:
        save_model_suffix = str(alpha)

    tester = PastaTester(model_name=model_name, 
                         template=template,
                         evaluation_method=evaluation_method, 
                         alpha=alpha, 
                         context_lengths_min=context_lengths_min, 
                         context_lengths_max=context_lengths_max,
                         start_context_lengths=start_context_lengths,
                         context_lengths=context_lengths,
                         document_depth_percents=document_depths,
                         needle_name=needle_name,
                         add_hint=add_hint,
                         save_model_suffix=save_model_suffix,
                         attn_implementation='eager')
    tester.start_test()

if __name__ == '__main__':
    CLI(main)
