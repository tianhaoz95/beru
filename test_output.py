from model.model import BeruModel
from model.model_config import BeruConfig
from tokenizers import Tokenizer
import torch
import utils


def generate_text(prompt: str, max_tokens: int = 100):
    config = BeruConfig(n_layers=2)
    model = BeruModel(config)
    checkpoint_path = (
        "/Users/tianhaoz/Downloads/beru/checkpoints/checkpoint-500/pytorch_model.bin"
    )
    device = torch.device(utils.get_device())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    tokenizer = Tokenizer.from_file("bpe-small-vocab-for-beru.json")
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids]).to(device)
    generated_sequences = []
    for output_ids in model.generate(
        input_ids, max_new_tokens=max_tokens, temperature=0.7, top_p=0.9, use_cache=True
    ):
        generated_sequences += output_ids.tolist()[0]
    generated_text = tokenizer.decode(generated_sequences)
    return generated_text


def main():
    prompt = "Once upon a time"
    print(f"Prompt: {prompt}")

    generated_text = generate_text(prompt)
    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main()
