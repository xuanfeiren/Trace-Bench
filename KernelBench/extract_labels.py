import json

# Read the JSON file
with open('dataset/cuda_level1_dataset_with_labels.json', 'r') as f:
    data = json.load(f)

# Extract all labels
labels = [item['label'] for item in data]

# Print statistics
print(f"Total entries: {len(data)}")
print(f"Total labels: {len(labels)}")
print("\n" + "="*80)
print("All labels:")
print("="*80 + "\n")

# Print all labels with index
for i, label in enumerate(labels, 1):
    print(f"{i}. {label}")
    print("-"*80)

# Save labels to a text file
with open('extracted_labels.txt', 'w') as f:
    f.write(f"Total entries: {len(data)}\n")
    f.write(f"Total labels: {len(labels)}\n\n")
    f.write("="*80 + "\n")
    f.write("All labels:\n")
    f.write("="*80 + "\n\n")
    for i, label in enumerate(labels, 1):
        f.write(f"{i}. {label}\n")
        f.write("-"*80 + "\n")

print(f"\n\nLabels saved to 'extracted_labels.txt'")
