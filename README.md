On the Loss of Context-awareness in General Instruction Finetuning
==============

[![arXiv](https://img.shields.io/badge/arXiv-2402.16459-b31b1b.svg)](http://arxiv.org/abs/2411.02688)

Pretrained Large Language Models (LLMs) require post-training methods such as supervised fine-tuning (SFT) on instruction-response pairs to enable instruction following. 
However, this process can potentially harm existing capabilities learned during pretraining. 
In this paper, we investigate the loss of context awareness after SFT, defined as the capability to extract and understand information from the user-provided context and respond accordingly. 
We are the first to identify and show that the loss of context-awareness appears on instruction-finetuned LLMs when the chat template is applied to the input prompts. 
We identify the performance decline is partially caused by the bias embedded into the chat template to focus less on the the user-provided context.
Based on these observations, we propose two methods to mitigate the loss of context awareness in instruct models: post-hoc attention steering on user prompts and conditional instruction fine-tuning with a context-dependency indicator.
Empirical experiments on 4 context-dependent downstream tasks and 3 pretrained LLMs of different sizes show that our methods can effectively mitigate the loss of context awareness without compromising the general ability of instruction following. 
Our findings also strongly advocate the necessity to benchmark context awareness after instruction fine-tuning carefully.

Bibtex for our [paper](http://arxiv.org/abs/2411.02688):
```bibtex
@misc{wang2024losscontextawarenessgeneralinstruction,
      title={On the loss of context-awareness in general instruction fine-tuning}, 
      author={Yihan Wang and Andrew Bai and Nanyun Peng and Cho-Jui Hsieh},
      year={2024},
      eprint={2411.02688},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.02688}, 
}
```

## Setup
* Install [alignment-handbook](https://github.com/huggingface/alignment-handbook)
* ```
    cd LLMTest_NeedleInAHaystack
    pip install -r requirements.txt
  ```
* ```
    cd lm-evaluation-harness
    pip install -r requirements.txt
    pip install -e .
  ```
* ```
    cd PASTA
    pip install -r requirements.txt
    pip install -e .
  ```

## Post-hoc Attention Steering on the User Context
### Wrapping a Huggingface Model
```
from UserSteeredWrapper import UserSteeredWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
from pastalib.utils.CustomLlama3Tokenizer import CustomLlama3Tokenizer


model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
tokenizer = CustomLlama3Tokenizer(tokenizer)

warnings.warn("Currently using hot-patched llama-3 tokenizer. Update to officiel tokenizer when offset_mapping bug is fixed.")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

wrapped_model = UserSteeredWrapper({k: None for k in range(32)}, model, tokenizer, alpha=0.9)

conv = [{'role': 'user', 'content': 'Hello'}]
output = wrapped_model.generate(conv)
print(tokenizer.decode(output, skip_special_tokens=True))
```

### Reproducing Experiments
**Evaluate on Needle-in-a-Haystack (NIH)**

```
cd LLMTest_NeedleInAHaystack
python PastaTester.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --needle_name SF --context_lengths_max 8000 --alpha 0.95 --save_model_suffix alpha_0.95
```

**Evaluate on lm-eval tasks**

```
lm_eval     --model hf-pasta     --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,dtype="float",use_chat_template=True,add_bos_token=True,parallelize=True,alpha=0.9     --task drop_no_ind    --device cuda:0 --batch_size 8
```


## Conditional Finetuning with Context-Dependency Indicators
### Reproducing the Finetuning
**Labeling instruction-finetuning dataset with context-dependency score**

```
python process_dataset.py --dataset_name ucla-cmllab/vicuna_cleaned --preprocess_function_name add_prompt --configs='{"probe_model":"ucla-cmllab/tinyllama-sft-vicuna-full"}' --save_huggingface_hub yihanwang617/vicuna_cleaned_processed
```
This script will append a context-dependency score to each user prompt, in the format of "The user attention ratio is [context-dependency-score]."
Pre-processed datasets with the context-dependency scores for WizardLM-70k, ShareGPT(Vicuna), and UltraChat-200k can be found in the [HF collection](https://huggingface.co/collections/ucla-cmllab/context-awareness-in-instruction-finetuning-671b44e2a9a89705ec2b8208)

```
python process_dataset.py --dataset_name yihanwang617/vicuna_cleaned_processed --preprocess_function_name modify_prompt --save_huggingface_hub yihanwang617/vicuna_cleaned_processed_indicator_0.6 --config='{"threshold":0.6}'
```
This script replaces the context-dependency string appended to each user prompt with a special character '[IND]' if the score is larger than the given threshold, and an empty string '' if it's lower than the threshold.

**Instruction finetuning with the indicators**

We finetune all the models mentioned in the paper following the recipes provided in [huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook) with some necessary modifications. Due to limited computational resources, we finetune Llama-3-8B and Llama-2-7B with QLora. All finetuning were run on 4 A6000 GPUs. 

To finetune TinyLlama on a processed ShareGPT(Vicuna) dataset with the indicator '[IND]', we can run
```
ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29500 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/tinyllama-v1.0/sft/config_full_vicuna_processed.yaml
```

To finetune Llama-3-8B or Llama-2-7B with QLora on a processed ShareGPT(Vicuna) dataset, we can run
```
ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29500 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/llama-2/config_qlora_vicuna_processed.yaml --load_in_4bit=false
```

```
ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29500 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/llama-3/config_qlora_vicuna_processed.yaml --load_in_4bit=false
```

### Evaluation
We have released a series of models finetuned with the the context-dependency indicator '[IND]' in this [HF collection](https://huggingface.co/collections/ucla-cmllab/context-awareness-in-instruction-finetuning-671b44e2a9a89705ec2b8208).


**Evaluation on Needle-in-a-Haystack**

We included a modified version of the Needle_in_a_Haystack in this repository.
To evaluate a model with NIH test:
```
cd LLMTest_NeedleInAHaystack
python HuggingfaceTester.py --model_name ucla-cmllab/tinyllama-sft-vicuna-processed-indicator-0.6 --needle_name {SF-indicator, SF} --context_lengths_max {context_lengths_max}
```

The above command will launch a NIH test with 400 tests in total, with 20 context lengths from 200 to {context_lengths_max} and 20 needle insertion depth from 0% to 100%. Results will be put into a results folder.

`needle_name` specifies the needle and the question used in NIH test. In `--needle_name SF-indicator`, a context-dependency indicator is appended to the question as "What is the best thing to do in San Francisco? [IND]"

To calculate the average recall error reported in the paper and visualize the recall error heatmap, run
```
python viz/CreateVizFromLLMTesting.py [path/to/the/results/folder] --max_length {context_lengths_max} --re_evaluation_method subword_match --needle_name SF
```

**Evaluation on lm-eval tasks**

We evaluate the models on contextual QA tasks (drop, squad, quac) under the lm-eval benchmark.

```
lm_eval     --model hf     --model_args pretrained=ucla-cmllab/tinyllama-sft-vicuna-processed-indicator-0.6,dtype="float",use_chat_template=True,add_bos_token=True,parallelize=True   --task {drop, drop_no_ind, quac, quac_no_ind, squad, squad_no_ind}    --device cuda:0 --batch_size 8
```

## Acknowledgement
We have referred to code from
* [gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
* [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* [QingruZhang/PASTA](https://github.com/QingruZhang/PASTA)
* [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)