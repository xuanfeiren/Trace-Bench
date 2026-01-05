import json
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from dataset.utils import load_entire_cuda_dataset

def main():
    print("Loading entire CUDA dataset...")
    dataset = load_entire_cuda_dataset()

    print(f"Loaded {len(dataset)} examples")

    # Save to dataset folder as JSON
    output_path = Path(__file__).parent / "dataset" / "cuda_level1_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved to {output_path}")
    print(f"\nDataset summary:")
    print(f"  - Total examples: {len(dataset)}")
    if dataset:
        print(f"  - Fields per example: {list(dataset[0].keys())}")
        print(f"\nFirst example preview:")
        print(f"  - Input length: {len(dataset[0]['input'])} characters")
        print(f"  - Ref arch src length: {len(dataset[0]['ref_arch_src'])} characters")

if __name__ == "__main__":
    main()
