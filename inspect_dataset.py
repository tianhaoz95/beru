from datasets import load_dataset
import utils
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("my_bpe_tokenizer_files")
    dataset = utils.chunk_and_tokenize_dataset(tokenizer)
    
    print("\n=== Dataset Information ===")
    print(f"Dataset features: {dataset.features}")
    
    print("\n=== First Few Examples ===")
    for i, example in enumerate(dataset.take(3)):
        print(f"\nExample {i + 1}:")
        print(f"Input IDs length: {len(example['input_ids'])}")
        print(f"Labels length: {len(example['labels'])}")
        
        decoded_text = tokenizer.decode(example['input_ids'][:50])
        print(f"First 50 tokens decoded: {decoded_text}")

if __name__ == "__main__":
    main()