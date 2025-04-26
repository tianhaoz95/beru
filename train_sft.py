import argparse

import torch
from datasets import load_dataset
from model.model import BeruModel
from model.model_config import BeruConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# from utils import chunk_and_tokenize_sft_dataset


def main():
    parser = argparse.ArgumentParser(description="Train a model using SFT")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=20,
        help="Number of steps between model checkpoints (default: 20)",
    )
    args = parser.parse_args()

    AutoConfig.register("beru", BeruConfig)
    AutoModelForCausalLM.register(BeruConfig, BeruModel)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    training_args = SFTConfig(
        max_length=512,
        output_dir="sft_model",
        max_steps=100,
        save_steps=10,
    )

    ds = load_dataset("philschmid/dolly-15k-oai-style", split="train", streaming=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model("./final_sft_model")


if __name__ == "__main__":
    main()
