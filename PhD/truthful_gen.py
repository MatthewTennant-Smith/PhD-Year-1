import pandas as pd

input_file = "validation-00000-of-00001.parquet"
output_file = "truthfulqa_validation.json"

df = pd.read_parquet(input_file)

df.to_json(
    output_file,
    orient="records",
    force_ascii=False,
    indent=2
)

print(f"Saved {len(df)} records to {output_file}")