from datasets import load_dataset
import numpy as np
import huggingface_hub

def convert(example):
    new_messages = []
    for msg in example["messages"]:
        if msg["from"] == "human":
            role = "user"
        elif msg["from"] == "gpt":
            role = "assistant"
        else:
            raise NotImplementedError
        new_messages.append({'role': role, 'content': msg['value']})
    example["messages"] = new_messages
    return example

def main():
    huggingface_hub.login()
    dset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k")['train']

    dset = dset.rename_column("conversations", "messages")
    dset = dset.map(convert, num_proc=32)

    dset.push_to_hub("ucla-cmllab/WizardLM_evol_instruct_V2_196k-chat-format", split='train', 
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
        dset_.push_to_hub("ucla-cmllab/WizardLM_evol_instruct_V2_100k-chat-format", 
                          split=split_, private=True)
        
        split_ = f"{split}"
        dset_.push_to_hub("ucla-cmllab/WizardLM_evol_instruct_V2_100k-chat-format", 
                          split=split_, private=True)

if __name__ == "__main__":
    main()

