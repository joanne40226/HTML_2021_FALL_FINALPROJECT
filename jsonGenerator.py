import os
import json
import random

dataset_root = "../formatted_dataset"

sequences = [seq for seq in os.listdir(dataset_root) if seq.startswith("sequence_")]

random.seed(42)  
random.shuffle(sequences)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

n_total = len(sequences)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_sequences = sequences[:n_train]
val_sequences = sequences[n_train:n_train + n_val]
test_sequences = sequences[n_train + n_val:]

output_dir = os.path.join(dataset_root, "train_test_split")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "shuffled_train_file_list.json"), "w") as f:
    json.dump(train_sequences, f, indent=4)

with open(os.path.join(output_dir, "shuffled_val_file_list.json"), "w") as f:
    json.dump(val_sequences, f, indent=4)

with open(os.path.join(output_dir, "shuffled_test_file_list.json"), "w") as f:
    json.dump(test_sequences, f, indent=4)

print("splitting succeed！")
print(f"training set: {len(train_sequences)}")
print(f"val set: {len(val_sequences)}")
print(f"testing set: {len(test_sequences)}")

