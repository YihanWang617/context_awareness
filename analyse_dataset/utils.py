import torch
import numpy as np


def conv_to_text(conversation, probe_tokenizer):
    conversation['text'] = probe_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    return conversation

def tokenize(examples, tokenizer, max_length):
    return tokenizer(examples["text"], return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)


def build_completion_mask(batch, system_token_ids, response_token_ids, instruction_token_ids):
    completion_mask = torch.ones(batch["input_ids"].shape[0], batch["input_ids"].shape[1])
    system_mask = torch.ones(batch["input_ids"].shape[0], batch["input_ids"].shape[1])
    completion_masks = [] # seperate masks for each response completion
    for i in range(len(batch["input_ids"])):
        response_token_ids_idxs = []
        human_token_ids_idxs = []
        system_token_ids_idxs = []

        for system_idx in np.where(batch["input_ids"][i] == system_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (
                system_token_ids
                == batch["input_ids"][i][system_idx : system_idx + len(system_token_ids)].tolist()
            ):
                system_token_ids_idxs.append(system_idx)
                system_mask[i][system_idx:system_idx + len(system_token_ids)] = 0

        for assistant_idx in np.where(batch["input_ids"][i] == response_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (
                response_token_ids
                == batch["input_ids"][i][assistant_idx : assistant_idx + len(response_token_ids)].tolist()
            ):
                response_token_ids_idxs.append(assistant_idx + len(response_token_ids))
                system_mask[i][assistant_idx:assistant_idx + len(response_token_ids)] = 0

        human_token_ids = instruction_token_ids
        for human_idx in np.where(batch["input_ids"][i] == human_token_ids[0])[0]:
            # find the indexes of the start of a human answer.
            if human_token_ids == batch["input_ids"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                human_token_ids_idxs.append(human_idx)
                system_mask[i][human_idx:human_idx + len(human_token_ids)] = 0

        if len(response_token_ids_idxs) == 0 or len(human_token_ids_idxs) == 0:
                completion_mask[i,:] = 0
        
        if (
            len(human_token_ids_idxs) > 0
            and len(response_token_ids_idxs) > 0
            and human_token_ids_idxs[0] > response_token_ids_idxs[0]
        ):
            human_token_ids_idxs = [0] + human_token_ids_idxs


        for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
            # Make pytorch loss function ignore all non response tokens
            if idx != 0:
                completion_mask[i, start:end] = 0
            else:
                completion_mask[i, :end] = 0

        for idx, (start, end) in enumerate(zip(response_token_ids_idxs[:-1], human_token_ids_idxs[1:])):
            response_mask = torch.zeros(batch["input_ids"].shape[0], batch["input_ids"].shape[1])
            response_mask[i,start:end] = 1
            completion_masks.append(response_mask)

        if len(response_token_ids_idxs) < len(human_token_ids_idxs):
            completion_mask[i, human_token_ids_idxs[-1] :] = 0
        if len(human_token_ids_idxs) > 0 and len(response_token_ids_idxs) > 0 and response_token_ids_idxs[-1] > human_token_ids_idxs[-1]:
            response_mask = torch.zeros(batch["input_ids"].shape[0], batch["input_ids"].shape[1])
            response_mask[i,response_token_ids_idxs[-1]:] = 1
            completion_masks.append(response_mask)

    return completion_mask, system_mask, completion_masks


def get_batch_user_attention(batch, probe_model, probe_tokenizer=None, layer_index = 15, head_list = list(range(32)), top_k = 1, return_separate=False, add_normalization=False, mask_template=False):
    batch_input_ids = batch['input_ids'].to('cuda')
    batch_attention_mask = batch['attention_mask'].to('cuda')

    completion_mask, system_mask, completion_masks = build_completion_mask(batch, system_token_ids=[1,529,29989,5205,29989,29958,13], response_token_ids = [29966, 29989, 465, 22137, 29989, 29958, 13], instruction_token_ids = [29966, 29989, 1792, 29989, 29958, 13])
    attention_mask = (batch_input_ids == 13) | (batch_input_ids == 1) | (batch_input_ids == 2)

    if mask_template:
        batch_attention_mask = batch_attention_mask & system_mask.to('cuda').to(torch.bool) & (~attention_mask.to('cuda').to(torch.bool))

    outputs = probe_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, output_attentions=True)  # Run model
    attention = outputs[-1][layer_index][:] # [batch_size, num_head, max_token, max_token]
    head_index = torch.LongTensor(head_list).to(attention.device)
    attention = attention.index_select(1, head_index)


    # attention *= ~attention_mask.unsqueeze(1).unsqueeze(1)
    max_length = attention.shape[-1]
    causal_mask = torch.Tensor(np.tril(np.ones((max_length, max_length), dtype=bool))).to(attention.device)
    attention *= causal_mask

    completion_mask = completion_mask.to(attention.device).to(torch.bool)
    system_mask = system_mask.to(attention.device).to(torch.bool)
    batch_attention_mask = batch_attention_mask.to(attention.device).to(torch.bool)

    # attention *= system_mask.view(system_mask.shape[0], 1, 1, system_mask.shape[-1]).to('cuda') # [batch_size, num_head, max_token, max_token]
    # assistant_mask = batch_attention_mask&completion_mask
    assistant_masks = [system_mask & batch_attention_mask &mask.to(attention.device).to(torch.bool) for mask in completion_masks]
    # assistant_mask = batch_attention_mask&completion_mask
    # if not ~((~assistant_mask) & (sum(assistant_masks))).all():
    #     import pdb; pdb.set_trace()
    # attention *= assistant_mask.unsqueeze(1).unsqueeze(-1)

    user_mask = (~completion_mask) & (system_mask) & (~attention_mask) # [batch_size, max_token]
    user_attention = attention * user_mask.view(user_mask.shape[0], 1, 1, user_mask.shape[-1])

    user_attention = user_attention.sum(-1).max(1)[0]

    # assistant_lengths = [mask.sum().item() for mask in assistant_masks]
    # user_lengths = [user_mask[:,:mask[0].nonzero()[0][0]].sum().item() if len(mask[0].nonzero()) > 0 else 0 for mask in assistant_masks]

    # total_lengths = [mask[0].nonzero()[-1][0].item() + 1 if len(mask[0].nonzero()) > 0 else 0 for mask in assistant_masks]
    # import pdb; pdb.set_trace()
    # normalization = [math.log((t_l)/(t_l - a_l)) * u_l/a_l  if a_l != 0 and t_l !=0 else -1 for a_l, u_l, t_l in zip(assistant_lengths, user_lengths, total_lengths)]
    top_user_attention_separate = [torch.masked_select(user_attention, mask) for mask in assistant_masks]

    top_user_attention_separate = [(att.sum()/att.shape[0]).item() for att in top_user_attention_separate]
    # if add_normalization:
    #     top_user_attention_separate = [0 if math.isnan(a) else a - n for a, n in zip(top_user_attention_separate, normalization) if n > 0]
    if len(top_user_attention_separate) == 0:
        return -1, [-1]
    else:
        top_user_attention = sum(top_user_attention_separate)/len(top_user_attention_separate)

    if return_separate:
        return top_user_attention, top_user_attention_separate
    return top_user_attention


def get_batch_length(batch, probe_model, probe_tokenizer):
    lengths = batch['attention_mask'].sum(dim=-1).to(torch.float)
    
    return lengths.mean().item()