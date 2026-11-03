import gzip
import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\matti\hh-rlhf")
OUTPUT_DIR = BASE_DIR / "evaluation"

OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    "helpful-base",
    "harmless-base",
]


def split_conversation(text):


    # Find the final Assistant response
    marker = "\n\nAssistant:"

    position = text.rfind(marker)

    if position == -1:
        raise ValueError("Could not find final Assistant message")

    prompt = text[:position]
    response = text[position + len(marker):]

    # Clean up whitespace
    prompt = prompt.strip()
    response = response.strip()

    return prompt, response


output_file = OUTPUT_DIR / "hh_rlhf_test.jsonl"

total = 0

with open(output_file, "w", encoding="utf-8") as out:

    for dataset in DATASETS:

        input_file = BASE_DIR / dataset / "test.jsonl.gz"

        print(f"Processing: {dataset}")

        with gzip.open(input_file, "rt", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                example = json.loads(line)

                # Extract prompt and responses
                chosen_prompt, preferred = split_conversation(
                    example["chosen"]
                )

                rejected_prompt, rejected = split_conversation(
                    example["rejected"]
                )

                # Make sure both versions have the same prompt
                if chosen_prompt != rejected_prompt:
                    print(
                        f"WARNING: prompt mismatch in {dataset} "
                        f"example {total}"
                    )

                result = {
                    "id": f"{dataset}-{total:06d}",
                    "source": dataset,
                    "prompt": chosen_prompt,
                    "preferred": preferred,
                    "rejected": rejected,
                }

                out.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )

                total += 1


print()
print("Finished!")
print(f"Total examples: {total}")
print(f"Output: {output_file}")