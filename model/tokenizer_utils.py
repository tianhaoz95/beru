import argparse
import os
import sys
from typing import List

from datasets import load_dataset
from model_config import BeruConfig
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import ByteLevel, WhitespaceSplit
from tokenizers.trainers import BpeTrainer
from transformers import PretrainedConfig, PreTrainedTokenizerFast


def batch_iterator(dataset, batch_size=100, total_batches=500):
    batch = []
    for _ in range(total_batches):
        examples = dataset.take(batch_size)
        for example in examples:
            text = example.get("text")
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def train_tokenizer(
    batches_to_train=10000,
    batch_size=20,
    output_dir="my_bpe_tokenizer_files",
    params: BeruConfig = BeruConfig(),
):
    ds = load_dataset(
        "HuggingFaceFW/fineweb", "CC-MAIN-2013-20", split="train", streaming=True
    )

    unk_token = "<|unknown|>"
    bos_token = "<|begin_of_text|>"
    eos_token = "<|end_of_text|>"
    pad_token = "<|pad|>"
    special_tokens_list = [
        bos_token,
        eos_token,
        unk_token,
        pad_token,
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",
    ] + [f"<|reserved_special_token_{i}|>" for i in range(10)]

    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = WhitespaceSplit()

    trainer = BpeTrainer(
        vocab_size=params.vocab_size,
        min_frequency=2,
        special_tokens=special_tokens_list,
    )

    print(f"Starting tokenizer training for {batches_to_train} batches...")
    tokenizer.train_from_iterator(
        batch_iterator(ds, batch_size=batch_size, total_batches=batches_to_train),
        trainer=trainer,
        length=batch_size * batches_to_train,
    )
    print("Training finished.")

    print("Wrapping trained tokenizer...")
    tokenizer_wrapper = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        pad_token=pad_token,
    )
    print("Tokenizer wrapped.")

    print(f"Saving tokenizer files to directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_wrapper.save_pretrained(output_dir)
    print(f"Tokenizer successfully saved to {output_dir}")

    print("\n--- Testing Loading with AutoTokenizer ---")
    try:
        from transformers import AutoTokenizer

        reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print("Successfully reloaded tokenizer using AutoTokenizer.")

        test_text = "Hello, y'all! How are you 😁?"
        print("Original test text:", test_text)

        encoded = reloaded_tokenizer.encode(test_text)
        print("Encoded tokens:", encoded)

        decoded_cleaned = reloaded_tokenizer.decode(
            encoded, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        print("Decoded text (cleaned):", decoded_cleaned)

        decoded_raw = reloaded_tokenizer.decode(
            encoded, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )
        print("Decoded text (raw):", decoded_raw)

    except Exception as e:
        print(f"Error testing AutoTokenizer loading: {e}")
