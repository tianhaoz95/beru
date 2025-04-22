import torch
from torch import nn
import sys
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Inspect PyTorch model parameters')
    parser.add_argument('--model', '-m', type=str, required=True,
                      help='Path to the model.bin file')
    args = parser.parse_args()

    model_path = Path(args.model)
    
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        total_params = sum(p.numel() for p in state_dict.values())
        
        print("\nModel Layers:")
        for key in state_dict.keys():
            print(f"Layer: {key}, Shape: {state_dict[key].shape}")

        print(f"\nModel Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Total parameters (in millions): {total_params/1e6:.2f}M")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()