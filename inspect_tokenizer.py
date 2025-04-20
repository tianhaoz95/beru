from transformers import PreTrainedTokenizerFast


def main():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe-small-vocab-for-beru.json")

    sample_text = (
        "Hello, this is a test sentence to check if the tokenizer is working properly."
    )
    encoded = tokenizer.encode(sample_text)
    print("Original text:", sample_text)
    print("Encoded tokens:", encoded)

    decoded = tokenizer.decode(
        encoded, clean_up_tokenization_spaces=True, skip_special_tokens=True
    )
    print("Decoded text:", decoded)


if __name__ == "__main__":
    main()
