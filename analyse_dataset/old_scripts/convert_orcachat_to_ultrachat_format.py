from datasets import load_dataset
import numpy as np
import huggingface_hub

def convert(example):
    new_messages = []
    assert len(example["messages"]) > 0
    new_messages.append({'role': 'system', 'content': example["instruction"]})

    for msg in example["messages"]:
        new_messages.append({'role': 'user', 'content': msg['input']})
        new_messages.append({'role': 'assistant', 'content': msg['output']})
    example["messages"] = new_messages
    return example

def main():
    huggingface_hub.login()
    dset = load_dataset("shahules786/orca-chat")['train']

    dset = dset.rename_column("conversation", "messages")
    n_prev = len(dset)
    dset = dset.filter(lambda example: len(example["messages"]) > 0)
    n_post = len(dset)
    print(f"Removed {n_prev - n_post} empty conversation examples.")
    dset = dset.map(convert, num_proc=32, remove_columns=["instruction"])

    dset.push_to_hub("ucla-cmllab/orca-chat-chat-format", split='train', 
                     private=False)

    iis = {}
    np.random.seed(1126)
    train_mask = np.zeros(len(dset), dtype=bool)
    iis['train'] = np.random.choice(len(dset), size=100000, replace=False)
    train_mask[iis['train']] = True
    iis['test'] = np.arange(len(dset))[~train_mask]

    assert len(np.intersect1d(iis['train'], iis['test'])) == 0

    for split in ['train', 'test']:
        dset_ = dset.select(iis[split])

        split_ = f"{split}_sft"
        dset_.push_to_hub("ucla-cmllab/orca-chat_100k-chat-format", 
                          split=split_, private=True)
        
        split_ = f"{split}"
        dset_.push_to_hub("ucla-cmllab/orca-chat_100k-chat-format", 
                          split=split_, private=True)

if __name__ == "__main__":
    main()

