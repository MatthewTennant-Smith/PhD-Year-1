import pandas as pd
import json
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "validation-00000-of-00001.parquet"
OUTPUT_FILE = "truthfulqa_validation.json"


# ============================================================
# Convert NumPy / PyArrow / pandas objects to JSON-compatible
# Python objects
# ============================================================

def make_json_serializable(value):

    # NumPy array -> Python list
    if isinstance(value, np.ndarray):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    # NumPy scalar -> Python scalar
    if isinstance(value, np.generic):
        return value.item()

    # Dictionary -> recursively convert values
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(val)
            for key, val in value.items()
        }

    # List / tuple -> recursively convert items
    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    # pandas NA / NaN handling
    if pd.isna(value):
        return None

    # Everything else
    return value


# ============================================================
# Main
# ============================================================

def main():

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not input_path.exists():
        print("ERROR: Input file not found:")
        print(input_path.resolve())
        return

    # --------------------------------------------------------
    # Read Parquet
    # --------------------------------------------------------

    print("Reading Parquet file...")

    df = pd.read_parquet(input_path)

    print(f"Found {len(df)} examples.")
    print()

    # --------------------------------------------------------
    # Show columns
    # --------------------------------------------------------

    print("Columns found:")

    for column in df.columns:
        print(f"  - {column}")

    print()

    # --------------------------------------------------------
    # Convert DataFrame to dictionaries
    # --------------------------------------------------------

    records = df.to_dict(orient="records")

    # --------------------------------------------------------
    # Recursively convert everything into JSON-compatible
    # Python objects
    # --------------------------------------------------------

    print("Converting NumPy/Pandas objects...")

    records = [
        make_json_serializable(record)
        for record in records
    ]

    # --------------------------------------------------------
    # Write JSON
    # --------------------------------------------------------

    print("Writing JSON file to:")
    print(f"  {output_path.resolve()}")

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Verify JSON
    # --------------------------------------------------------

    print()
    print("Verifying JSON file...")

    with open(
        output_path,
        "r",
        encoding="utf-8"
    ) as f:

        loaded_data = json.load(f)

    print(f"Successfully loaded {len(loaded_data)} examples.")

    # --------------------------------------------------------
    # Print first example
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST EXAMPLE")
    print("=" * 70)

    print(
        json.dumps(
            loaded_data[0],
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print("Created:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
