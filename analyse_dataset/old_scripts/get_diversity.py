import os

from task2vec import Task2Vec, Embedding
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from utils import conv_to_text
from functools import partial
from glob import glob
from argparse import ArgumentParser
from tqdm.auto import trange
import task_similarity

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--embs_dir', type=str)
    parser.add_argument('--save_fig_path', type=str)
    parser.add_argument('--pdist_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default="yihanwang617/vicuna_cleaned")
    parser.add_argument('--split', type=str, default="train")

    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--conv_template_name', type=str, default="TinyLlama")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
# probe_network = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T").to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    probe_network = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda')

    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.map(partial(conv_to_text, template=args.conv_template_name))

    def preprocess(examples):
        return tokenizer(examples["text"], return_tensors="pt", padding='max_length', 
                         truncation=True, max_length=1024)

    columns = dataset.column_names

    os.makedirs(args.embs_dir, exist_ok=True)

    embeddings = []
    for batch_num in trange(args.num_batches, desc="Embedding batches: "):
        print(f'--> {batch_num=}\n')

        emb_path = os.path.join(args.embs_dir, f"emb_{batch_num}.npz")
        if os.path.isfile(emb_path):
            embedding = Embedding.load(emb_path)
        else:
            shuffled_dataset = dataset.shuffle(seed=args.seed)
            batch = shuffled_dataset.select(range(args.batch_size))
            tokenized_batch = batch.map(preprocess, batched=True, remove_columns=columns)
            tokenized_batch.set_format(type="torch", columns=['input_ids', 'attention_mask'])
            embedding = Task2Vec(probe_network).embed(tokenized_batch)[0]
            embedding.save(emb_path)

        # import pdb; pdb.set_trace()
        embeddings.append(embedding)

    if (args.pdist_path is None) or (not os.path.isfile(args.pdist_path)):
        distance_matrix = task_similarity.pdist(embeddings, distance='cosine')
        if args.pdist_path is not None:
            np.save(args.pdist_path, distance_matrix)
    else:
        print(f"Loading distance matrix cached at {args.pdist_path}.")
        distance_matrix = np.load(args.pdist_path)

    div_coeff, conf_interval = task_similarity.stats_of_distance_matrix(distance_matrix)
    print(f"div_coeff: {div_coeff}")
    print(f"conf_interval: {conf_interval}")

    os.makedirs(os.path.dirname(args.save_fig_path), exist_ok=True)
    task_similarity.plot_distance_matrix(embeddings, show_plot=False, 
                                         save_fig=args.save_fig_path)

if __name__ == '__main__':
    main()

