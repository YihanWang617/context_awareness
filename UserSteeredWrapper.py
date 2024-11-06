"""
A wrapper class to wrap a Huggingface Model chat model and boosts the attention weights on user input.
"""
from pastalib.pasta import PASTA

class UserSteeredWrapper:
    def __init__(self, layer_head_dict, model, tokenizer, alpha=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_head_dict = layer_head_dict
        self.alpha = alpha


    def generate(self, conversation_history, contexts=None, **kwargs): # conversation should be in the format of [{'role': 'user', 'content': ...}]
        if type(conversation_history) == list:
            prompt = self.tokenizer.apply_chat_template(conversation_history, add_generation_prompt=True, tokenize=False)
            emphasized_texts = [a['content'] for a in conversation_history if a['role'] == 'user']
        else:
            prompt = conversation_history
            emphasized_texts = contexts

        pasta = PASTA(
            model=self.model,
            tokenizer=self.tokenizer,
            head_config=self.layer_head_dict, 
            alpha=self.alpha,
            scale_position="exclude",
        )

        inputs, offset_mapping = pasta.inputs_from_batch(prompt)

        with pasta.apply_steering(
            model=self.model, 
            strings=prompt, 
            substrings=emphasized_texts, 
            model_input=inputs,
            offsets_mapping=offset_mapping
        ) as steered_model: 
            output = steered_model.generate(**inputs, **kwargs)[0]

        return output[inputs.input_ids.shape[1]:]