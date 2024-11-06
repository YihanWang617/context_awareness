'''
Compute the scoring agreement between different layers in TinyLlama.
'''
from datasets import load_dataset
import math
import matplotlib.pyplot as plt
import numpy as np

layers = list(range(0, 22))
layer_sorts = {}
layer_scores = {}
for layer in layers:
    dataset_name = f"yihanwang617/vicuna_clean_processed_layer_{layer}"
    dataset = load_dataset(dataset_name)
    scores = []

    for i, item in enumerate(dataset['train']):
        messages = item['messages']
        for j, conv_msg in enumerate(messages):
            if "The user attention ratio is " in conv_msg['content']:
                conv_msg['content'], prompt = " ".join(conv_msg['content'].split(' ')[:-6]), conv_msg['content'].split(' ')[-6:]
                ratio = float(prompt[-1][:-1])
                if math.isnan(ratio):
                    ratio = -1
                scores.append((ratio, f"{i}_{j}"))
    layer_scores[layer] = [a[0] for a in scores]
    scores = sorted(scores)[len(scores)//10:]
    layer_sorts[layer] = [a[1] for a in scores]

# def larger(id_a, id_b, sorted_list):
#     if sorted_list.index(id_a) > sorted_list.index(id_b):
#         return True
    
mses = []
agreements = []
for layer_a in layers:
    mses.append([])
    agreements.append([])
    for layer_b in layers:
        score_a = layer_scores[layer_a]
        score_b = layer_scores[layer_b]
        
        orders_a = layer_sorts[layer_a]
        orders_b = layer_sorts[layer_b]

        mse = math.sqrt(sum([(a - b)**2 for a, b in zip(score_a, score_b)])/len(score_a))
        mses[-1].append(mse)

        agreement_cnt = 0
        disagreement_cnt = 0

        for i, id_a in enumerate(orders_a):
            if id_a not in orders_b:
                disagreement_cnt += 1
            else:
                agreement_cnt += 1
        agreements[-1].append(disagreement_cnt/(disagreement_cnt + agreement_cnt))


fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(agreements)

ax.set_xticks(np.arange(len(layers)), labels=[f'layer_{i}' for i in layers], fontsize=15)
ax.set_yticks(np.arange(len(layers)), labels=[f'layer_{i}' for i in layers], fontsize=15)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(layers)):
    for j in range(len(layers)):
        text = ax.text(j, i, '%.2f' % agreements[i][j],
                       ha="center", va="center", color="w", fontsize=12)

fig.tight_layout()
fig.savefig('agreements.pdf')



