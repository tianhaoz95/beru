import argparse
import sys
from typing import List
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import normalizers
from transformers import PretrainedConfig


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


def train_tokenizer(batches_to_train=10000, batch_size=20):
    ds = load_dataset(
        "HuggingFaceFW/fineweb", "CC-MAIN-2013-20", split="train", streaming=True
    )
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=6400,
        min_frequency=2,
        special_tokens=[
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
        ]
        + [f"<|reserved_special_token_{i}|>" for i in range(10)],
    )
    tokenizer.train_from_iterator(
        batch_iterator(ds, batch_size=batch_size, total_batches=batches_to_train),
        trainer=trainer,
        length=batch_size * batches_to_train,
    )
    tokenizer.save("bpe-small-vocab-for-beru.json")
    print(
        "Hello, y'all! How are you 😁 ? encoded to",
        tokenizer.encode("Hello, y'all! How are you 😁 ?").tokens,
    )
