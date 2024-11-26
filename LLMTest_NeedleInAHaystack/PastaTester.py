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
    
    # PASTA_HEAD_CONFIGS = {
    #     "tinyllama": {k: None for k in range(22)},
    #     "llama-2": {k: None for k in range(32)},
    #     "llama-3": {k: None for k in range(32)},
    # }

    # PASTA_HEAD_CONFIGS = {
    #     "tinyllama": {k: None for k in range(22)},
    #     "llama-2": {k: None for k in range(32)},
    #     "llama-3": {0: [21, 9, 8, 11, 15], 1: [ 9, 12, 28, 29, 8], 2: [ 6, 14, 19, 22, 15], 3: [ 9, 20, 22, 21, 0], 4: [15, 10, 1, 19, 16], 5: [17, 7, 1, 24, 4], 6: [31, 26, 25, 9, 6], 7: [26, 13, 15, 22, 25], 8: [ 8, 1, 9, 31, 11], 9: [15, 16, 2, 12, 3], 10: [14, 29, 1, 31, 13], 11: [18, 6, 15, 9, 13], 12: [ 9, 5, 21, 10, 15], 13: [ 5, 21, 8, 4, 18], 14: [29, 5, 12, 31, 22], 15: [11, 21, 26, 29, 30], 16: [ 0, 19, 25, 26, 1], 17: [31, 27, 26, 24, 29], 18: [ 9, 0, 8, 22, 20], 19: [23, 13, 3, 14, 9], 20: [30, 9, 26, 12, 14], 21: [ 8, 11, 1, 14, 26], 22: [10, 11, 27, 29, 8], 23: [19, 20, 25, 27, 22], 24: [24, 23, 3, 18, 27], 25: [17, 18, 2, 6, 5],26: [27, 15, 6, 19, 30],27: [ 4, 5, 28, 7, 16], 28: [15, 20, 23, 18, 0], 29: [31, 8, 11, 9, 16], 30:[17, 21, 18, 27, 2], 31:[30, 28, 7, 29, 21]},
    # }
    PASTA_HEAD_CONFIGS = {
        "tinyllama": {k: None for k in range(22)},
        "llama-2": {k: None for k in range(32)},
        "llama-3": {0: [15], 1: [8], 2: [15], 3: [0], 4: [16], 5: [4], 6: [6], 7: [25], 8: [11], 9: [3], 10: [13], 11: [13], 12: [15], 13: [18], 14: [22], 15: [30], 16: [1], 17: [29], 18: [20], 19: [9], 20: [14], 21: [26], 22: [8], 23: [22], 24: [27], 25: [5],26: [30],27: [6], 28: [0], 29: [16], 30:[2], 31:[21]},
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

    def generate(self, conversation_history, context=None, max_new_tokens=60):
        # print(prompt)
        
        output = self.wrapped_model.generate(conversation_history, context, max_new_tokens=max_new_tokens)
        
        return output

    async def get_response_from_model(self, prompt_context):
        # generate_ids = self.model_to_test.generate(inputs.input_ids, max_new_tokens=60, pad_token_id=self.tokenizer.pad_token_id)
        # generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        prompt, context = prompt_context
        generate_ids = self.generate(prompt, context, max_new_tokens=60)
        response = self.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return response

    def get_prompt(self, context, return_context=True):
        prompt = super().get_prompt(context)
        if return_context:
            return prompt, context
        else:
            return prompt

def main(model_name: str=None, template: str='default', needle_name: str="SF", evaluation_method: str="substring_match", 
         alpha: float=0.01, context_lengths_min: int=200, context_lengths_max: int=4000, 
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
                         context_lengths=context_lengths,
                         document_depth_percents=document_depths,
                         needle_name=needle_name,
                         add_hint=add_hint,
                         save_model_suffix=save_model_suffix,
                         attn_implementation='eager')
    tester.start_test()

if __name__ == '__main__':
    CLI(main)
