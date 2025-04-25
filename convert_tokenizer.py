import argparse

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def add_chat_template(tokenizer_path: str, output_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    chat_template = """
    {% for message in messages %}
        {% if message['role'] == 'user' %}
            {{ '<|reserved_special_token_0|>' + 'User: ' + message['content'] + '<|reserved_special_token_1|>' }}
        {% elif message['role'] == 'assistant' %}
            {{ '<|reserved_special_token_0|>' + 'Assistant: ' + message['content'] + '<|reserved_special_token_1|>' }}
        {% elif message['role'] == 'system' %}
            {{ '<|reserved_special_token_0|>' + 'System: ' + message['content'] + '<|reserved_special_token_1|>' }}
        {% endif %}
    {% endfor %}
    """
    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(description="Add chat template to a tokenizer")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input tokenizer"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save modified tokenizer"
    )

    args = parser.parse_args()
    add_chat_template(args.input, args.output)


if __name__ == "__main__":
    main()
