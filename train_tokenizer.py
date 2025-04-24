import argparse

from model.model_config import BeruConfig
from model.tokenizer_utils import train_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train tokenizer with specified parameters"
    )
    parser.add_argument(
        "--batches_to_train",
        type=int,
        default=100,
        help="Number of batches to train (default: 100)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Size of each batch (default: 100)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Size of vocabulary (default: 32000)",
    )

    args = parser.parse_args()

    params = BeruConfig(vocab_size=args.vocab_size)

    train_tokenizer(
        batches_to_train=args.batches_to_train,
        batch_size=args.batch_size,
        params=params,
    )


if __name__ == "__main__":
    main()
