from datasets import load_dataset
import huggingface_hub

def main():
    huggingface_hub.login()
    dset = load_dataset("habanoz/oasst_top1_2023-08-25-chat-format")

    for split in dset.keys():
        dset_ = dset[split].rename_column("conversation", "messages")
        split_ = f"{split}_sft"
        dset_.push_to_hub("ucla-cmllab/oasst_top1_2023-08-25-chat-format", split=split_, private=True)
        
        split_ = f"{split}"
        dset_.push_to_hub("ucla-cmllab/oasst_top1_2023-08-25-chat-format", split=split_, private=True)

if __name__ == "__main__":
    main()
