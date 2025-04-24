from model.model import BeruModel
from model.model_config import BeruConfig
from transformers import AutoTokenizer
from torch import optim, nn
import math
from transformers import Trainer, TrainingArguments
import torch
import argparse
from utils import chunk_and_tokenize_dataset
import utils

from transformers import TrainerCallback, TrainingArguments, Trainer


def train_model(
    max_steps,
    save_steps,
    checkpoint_path=None,
    generate_every=100,
    batch_size=100,
    tokenizer_path="my_bpe_tokenizer_files",
    vocab_size=32000,
):
    config = BeruConfig(vocab_size=vocab_size)
    model = BeruModel(config)
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ds = chunk_and_tokenize_dataset(tokenizer)

    text_to_encode = "This is a test sentence from the dataset."
    encoded = tokenizer.encode(text_to_encode)
    print(f"\n--- Testing Tokenizer ---")
    print(f"Original: {text_to_encode}")
    print(f"Encoded IDs: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=100,
        save_strategy="steps",
        save_steps=save_steps,
        save_safetensors=False,
        report_to="none",
    )

    class GenerationCallback(TrainerCallback):
        def __init__(self, tokenizer, generate_every=100):
            self.tokenizer = tokenizer
            self.generate_every = generate_every
            self.prompt = "This is a story about basketball"

        def on_step_end(self, args, state, control, model, **kwargs):
            if state.global_step % self.generate_every == 0:
                model.eval()
                with torch.no_grad():
                    encoded = tokenizer.encode(self.prompt)
                    device = torch.device(utils.get_device())
                    input_ids = torch.tensor([encoded]).to(device)
                    generated_sequences = []
                    print(f"\nStep {state.global_step} generation:")
                    print(f"Prompt: {self.prompt}")
                    for output_ids in model.generate(
                        input_ids,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=20,
                        temperature=0.7,
                        top_p=0.9,
                        use_cache=True,
                    ):
                        generated_sequences = output_ids.tolist()
                        current_text = tokenizer.decode(generated_sequences[0])
                        print(
                            f"\rGenerated: {self.prompt} {current_text}",
                            end="",
                            flush=True,
                        )
                    print("\n")
                model.train()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        callbacks=[GenerationCallback(tokenizer, generate_every=generate_every)],
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
    parser.add_argument(
        "--generate_every",
        type=int,
        default=100,
        help="Generate sample text every N steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per device training batch size",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="my_bpe_tokenizer_files",
        help="Path to the tokenizer files",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Size of the vocabulary (default: 32000)",
    )
    args = parser.parse_args()

    train_model(
        args.max_steps,
        args.save_steps,
        args.checkpoint_path,
        args.generate_every,
        args.batch_size,
        args.tokenizer_path,
        args.vocab_size,
    )


if __name__ == "__main__":
    main()
