import json
import random
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("./UltraFeedback")
OUTPUT_FILE = "ultrafeedback_eval_1200.json"

SAMPLES_PER_SOURCE = 200
SEED = 42

FILES = {
    "truthful_qa": "truthful_qa.jsonl",
    "false_qa": "false_qa.jsonl",
    "evol_instruct": "evol_instruct.jsonl",
    "ultrachat": "ultrachat.jsonl",
    "flan": "flan.jsonl",
    "sharegpt": "sharegpt.jsonl",
}


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as e:

                print(
                    f"Could not parse {path} "
                    f"line {line_number}: {e}"
                )

    return records


# ============================================================
# Extract instruction
# ============================================================

def get_instruction(record):

    instruction = record.get("instruction")

    if instruction is None:
        raise ValueError(
            "Could not find 'instruction' field:\n"
            + str(record)[:1000]
        )

    return instruction


# ============================================================
# Main
# ============================================================

random.seed(SEED)

evaluation_set = []

for source, filename in FILES.items():

    path = DATA_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find {path}"
        )

    records = load_jsonl(path)

    print(
        f"{source}: "
        f"{len(records)} records available"
    )

    if len(records) < SAMPLES_PER_SOURCE:

        raise ValueError(
            f"{source} only has {len(records)} records."
        )

    # --------------------------------------------------------
    # Select 200 examples
    # --------------------------------------------------------

    selected = random.sample(
        records,
        SAMPLES_PER_SOURCE
    )

    # --------------------------------------------------------
    # Keep original UltraFeedback information
    # --------------------------------------------------------

    for i, record in enumerate(selected):

        evaluation_example = {

            "id": f"{source}_{i:04d}",

            "source": source,

            # The question/instruction
            "instruction": get_instruction(record),

            # ------------------------------------------------
            # IMPORTANT:
            # Preserve the original UltraFeedback data.
            # This contains the human annotations/scores.
            # ------------------------------------------------

            "human_reference": {

                "models": record.get(
                    "models",
                    []
                ),

                "completions": record.get(
                    "completions",
                    []
                )
            }
        }

        evaluation_set.append(
            evaluation_example
        )


# ============================================================
# Shuffle
# ============================================================

random.shuffle(evaluation_set)


# ============================================================
# Save
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_set,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("UltraFeedback evaluation set created")
print("=" * 60)

print(
    f"Total examples: {len(evaluation_set)}"
)

print(
    f"Samples per source: {SAMPLES_PER_SOURCE}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print()

for source in FILES:

    count = sum(
        1
        for x in evaluation_set
        if x["source"] == source
    )

    print(
        f"{source:20s}: {count}"
    )