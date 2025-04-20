from datasets import load_dataset
from tokenizers import Tokenizer
import utils

def main():
    # Load tokenizer
    tokenizer = Tokenizer.from_file("bpe-small-vocab-for-beru.json")
    
    # Get processed dataset using the utility function
    dataset = utils.chunk_and_tokenize_dataset(tokenizer)
    
    # Print dataset information
    print("\n=== Dataset Information ===")
    print(f"Dataset features: {dataset.features}")
    
    # Get and print first few examples
    print("\n=== First Few Examples ===")
    for i, example in enumerate(dataset.take(3)):
        print(f"\nExample {i + 1}:")
        print(f"Input IDs length: {len(example['input_ids'])}")
        print(f"Labels length: {len(example['labels'])}")
        
        # Decode a portion of the input to show the actual text
        decoded_text = tokenizer.decode(example['input_ids'][:50])
        print(f"First 50 tokens decoded: {decoded_text}")

if __name__ == "__main__":
    main()