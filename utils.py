import torch
from datasets import load_dataset


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def chunk_and_tokenize_dataset(tokenizer, chunk_size=513):
    def tokenize_and_chunk(examples):
        batch = {"input_ids": [], "labels": []}
        for text in examples["text"]:
            tokens = tokenizer.encode(text).ids
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += [tokenizer.token_to_id("[PAD]")] * (
                        chunk_size - len(chunk)
                    )
                batch["input_ids"].append(chunk[:-1])
                batch["labels"].append(chunk[1:])
        return batch

    ds = load_dataset(
        "HuggingFaceFW/fineweb", "CC-MAIN-2013-20", split="train", streaming=True
    )
    print("Original dataset:", ds)
    mapped_ds = ds.map(
        tokenize_and_chunk,
        remove_columns=ds.column_names,
        batched=True,
    )
    print("Mapped dataset:", mapped_ds)
    return mapped_ds
