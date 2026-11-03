import gzip
import json

input_file = "test.jsonl.gz"
output_file = "test.json"

records = []

with gzip.open(input_file, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        records.append({
            "prompt": item["prompt"],
            "response": item["response"],
            "is_safe": bool(item["is_safe"])
        })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Converted {len(records)} examples to {output_file}")