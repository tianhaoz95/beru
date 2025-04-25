import argparse
import json

from transformers import AutoTokenizer


def inspect_chat_template(tokenizer_path, messages):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if not hasattr(tokenizer, "chat_template"):
            print(f"Warning: No chat template found for {tokenizer_path}")
            return

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print("\n=== Chat Template Output ===")
        print(f"Model: {tokenizer_path}")
        print("\nFormatted Text:")
        print("-" * 50)
        print(formatted)
        print("-" * 50)

        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        print("\nToken Count:", len(tokens))

    except Exception as e:
        print(f"Error occurred: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect chat template output for a model"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )

    args = parser.parse_args()

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]

    inspect_chat_template(args.tokenizer_path, messages)


if __name__ == "__main__":
    main()
