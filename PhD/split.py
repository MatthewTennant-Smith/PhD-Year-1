import json
import random
from pathlib import Path

# Get the folder where THIS script is located
folder = Path(__file__).resolve().parent

print("Script folder:", folder)

# Look for the JSON in the same folder
input_file = folder / "truthfulqa_generation.json"

print("Looking for:", input_file)
print("File exists:", input_file.exists())

if not input_file.exists():
    print("\nERROR: beavertails.json was not found!")
    print("Files in this folder:")
    for file in folder.iterdir():
        print("  ", file.name)
    input("Press Enter to exit...")
    exit()

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\nLoaded {len(data)} records.")

# Shuffle
random.seed(42)
random.shuffle(data)

# 80/20 split
split_index = int(len(data) * 0.8)

train_data = data[:split_index]
test_data = data[split_index:]

# Save files
train_file = folder / "truthfulqa_generation_train.json"
test_file = folder / "truthful_generation_test.json"

with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"Train: {len(train_data)}")
print(f"Test:  {len(test_data)}")
print(f"\nCreated:")
print(train_file)
print(test_file)

input("\nPress Enter to exit...")