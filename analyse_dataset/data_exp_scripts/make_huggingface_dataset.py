import json
import os
from datasets import DatasetDict, Dataset

directory = "/home/yihan/finetuning/LLMTest_NeedleInAHaystack/results/yihanwang617/llama-2-qlora-wizard-processed-indicator-0.6_SF-indicator_TinyLlama_False"
files = os.listdir(directory)
dataset = []
# import pdb; pdb.set_trace()
for file in files:
    if file.endswith('.json'):
        file_path = os.path.join(directory, file)
        item = json.load(open(file_path, 'r'))
        context = item['context']
        context_length = item['context_length']
        answer = 'The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.'
        conv = [{'role': 'user', 'content': f"You are a helpful AI assistant that answers a question using only the provided document: \n{context}\n\nQuestion: What is the best thing to do in San Francisco?"},
                {'role': 'assistant', 'content': answer}]
        dataset.append({'messages': conv})
# import pdb; pdb.set_trace()
dataset = Dataset.from_list(dataset)
dataset = DatasetDict({'train': dataset})
dataset.push_to_hub('yihanwang617/needle-in-a-haystack-1k')

