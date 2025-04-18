from model.model import BeruModel
from model.model_config import BeruConfig
from tokenizers import Tokenizer
from datasets import load_dataset
from torch import optim, nn
import math
from transformers import Trainer, TrainingArguments
import torch


def get_dataset(tokenizer):
    def tokenize_and_chunk(example):
        tokens = tokenizer.encode(example["text"]).ids
        chunk = tokens[: min(513, len(tokens))]
        if len(chunk) < 513:
            chunk += [tokenizer.token_to_id("[PAD]")] * (513 - len(chunk))
        example["input_ids"] = chunk[:-1]
        example["labels"] = chunk[1:]
        return example

    ds = load_dataset(
        "HuggingFaceFW/fineweb", "CC-MAIN-2013-20", split="train", streaming=True
    )
    print("Original dataset:", ds)
    mapped_ds = ds.map(
        tokenize_and_chunk,
        remove_columns=ds.column_names,
    )
    print("Mapped dataset:", mapped_ds)
    return mapped_ds


def train_model():
    config = BeruConfig(n_layers=2)
    model = BeruModel(config)
    print(model)

    tokenizer = Tokenizer.from_file("bpe-small-vocab-for-beru.json")
    ds = get_dataset(tokenizer)

    text_to_encode = "This is a test sentence from the dataset."
    encoded = tokenizer.encode(text_to_encode)
    print(f"\n--- Testing Tokenizer ---")
    print(f"Original: {text_to_encode}")
    print(f"Encoded IDs: {encoded.ids}")
    print(f"Encoded Tokens: {encoded.tokens}")
    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded: {decoded}")

    ds = get_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir="/Users/tianhaoz/Downloads/beru/checkpoints",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=10000,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
    )

    trainer.train()
    trainer.save_model(
        "/Users/tianhaoz/Downloads/beru/final_model", safe_serialization=False
    )


def main():
    train_model()


if __name__ == "__main__":
    main()
