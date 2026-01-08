import json
from tqdm import tqdm
import os

# Path to the downloaded file from the cache
# Using os.path.expanduser to handle the '~' correctly
cache_dir = os.path.expanduser('~/.cache/huggingface/hub/datasets--zifeng-ai--LEADSInstruct/snapshots/4346761bdf21545a367959134fec26e19433e64a/extraction/result_extraction/')
source_file_path = os.path.join(cache_dir, 'train.jsonl')

# Path for the new clean file in the current project directory
cleaned_file_path = 'cleaned_train.jsonl'

# Columns that might have mixed types and should be converted to strings
stringify_cols = ['denomValue', 'numValue', 'pmid']

print(f"Starting to clean {source_file_path}...")

# Read the large file line by line and write to the new file incrementally
try:
    with open(source_file_path, 'r', encoding='utf-8') as f_in, open(cleaned_file_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Cleaning and writing lines"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Enforce string type on problematic columns
                for col in stringify_cols:
                    if col in data and data[col] is not None:
                        data[col] = str(data[col])
                
                # Write the cleaned JSON object to the new file immediately
                f_out.write(json.dumps(data) + '\n')

            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line: {line.strip()}")

    print(f"Cleaning complete. Cleaned data saved to {cleaned_file_path}")

except FileNotFoundError:
    print(f"Error: Source file not found at {source_file_path}")
    print("Please ensure the dataset has been downloaded by a previous run.")
