from model.model import BeruModel
from model.model_config import BeruConfig
from tokenizers import Tokenizer
from datasets import load_dataset
from torch import optim, nn
import math
from transformers import Trainer, TrainingArguments
import torch
import argparse

from transformers import TrainerCallback, TrainingArguments, Trainer


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


def train_model(max_steps, save_steps, checkpoint_path=None):
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
        output_dir="./checkpoints",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=100,
        save_strategy="steps",
        save_steps=save_steps,
        save_safetensors=False,
    )

    class GenerationCallback(TrainerCallback):
        def __init__(self, tokenizer, generate_every=100):
            self.tokenizer = tokenizer
            self.generate_every = generate_every
            self.prompt = "The quick brown fox"

        def on_step_end(self, args, state, control, model, **kwargs):
            if state.global_step % self.generate_every == 0:
                model.eval()
                with torch.no_grad():
                    encoded = tokenizer.encode(self.prompt)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    input_ids = torch.tensor([encoded.ids]).to(device)
                    generated_sequences = []
                    for output_ids in model.generate(
                        input_ids,
                        max_new_tokens=20,
                        temperature=0.7,
                        top_p=0.9,
                        use_cache=True
                    ):
                        generated_sequences += output_ids.tolist()[0]
                    generated_text = tokenizer.decode(generated_sequences)
                    print(f"\nStep {state.global_step} generation:")
                    print(f"Prompt: {self.prompt}")
                    print(f"Generated: {generated_text}\n")
                model.train()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        callbacks=[GenerationCallback(tokenizer)],
    )

    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_model(
        "/Users/tianhaoz/Downloads/beru/final_model", safe_serialization=False
    )


def main():
    parser = argparse.ArgumentParser(description="Train the Beru model")
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    train_model(args.max_steps, args.save_steps, args.checkpoint_path)


if __name__ == "__main__":
    main()
