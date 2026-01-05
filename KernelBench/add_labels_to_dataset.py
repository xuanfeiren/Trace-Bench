import json
import ast
from pathlib import Path

def extract_model_docstring(source_code):
    """
    Extract the docstring from the Model class in the given source code.

    Args:
        source_code (str): Python source code containing a Model class

    Returns:
        str: The docstring of the Model class, or empty string if not found
    """
    try:
        tree = ast.parse(source_code)

        # Find the Model class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Model':
                # Get the docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    return docstring.strip()
                else:
                    return ""

        return ""
    except Exception as e:
        print(f"Error parsing code: {e}")
        return ""

def main():
    # Load the dataset
    dataset_path = Path(__file__).parent / "dataset" / "cuda_level1_dataset.json"
    print(f"Loading dataset from {dataset_path}")

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Processing {len(dataset)} examples...")

    # Add labels to each entry
    for i, entry in enumerate(dataset):
        ref_arch_src = entry['ref_arch_src']
        label = extract_model_docstring(ref_arch_src)
        entry['label'] = label

        if i < 5:  # Show first 5 examples
            print(f"\n=== Example {i} ===")
            print(f"Label: {label}")

    # Count how many entries have labels
    labeled_count = sum(1 for entry in dataset if entry['label'])
    print(f"\n\nSummary:")
    print(f"  - Total examples: {len(dataset)}")
    print(f"  - Examples with labels: {labeled_count}")
    print(f"  - Examples without labels: {len(dataset) - labeled_count}")

    # Save the updated dataset
    output_path = Path(__file__).parent / "dataset" / "cuda_level1_dataset_with_labels.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nUpdated dataset saved to {output_path}")

    # Also update the original file
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Original dataset file also updated with labels")

if __name__ == "__main__":
    main()
