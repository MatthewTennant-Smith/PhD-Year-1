import json
import random
from pathlib import Path

folder = Path(__file__).resolve().parent

input_file = folder / "hh_rlhf.jsonl"

print("Looking for:", input_file)
print("File exists:", input_file.exists())

# Load JSONL
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(data)} records.")

# Shuffle reproducibly
random.seed(42)
random.shuffle(data)

# 80/20 split
split_index = int(len(data) * 0.8)

train_data = data[:split_index]
test_data = data[split_index:]

# Save as JSONL
train_file = folder / "hh_rlhf_train.jsonl"
test_file = folder / "hh_rlhf_test.jsonl"

with open(train_file, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(test_file, "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Train: {len(train_data)}")
print(f"Test:  {len(test_data)}")
print(f"Created: {train_file}")
print(f"Created: {test_file}")