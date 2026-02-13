# prepare_data.py - Sample and prepare PII dataset for training

import json
import argparse
import random
from pathlib import Path


def prepare_dataset(input_path, output_path, max_records=50000, seed=42):
    """
    Sample records from the full dataset for training.

    Args:
        input_path: Path to output_normalized.jsonl (full dataset)
        output_path: Path for sampled output
        max_records: Number of records to sample
        seed: Random seed for reproducibility
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Load all records
    print(f"Loading data from {input_path}...")
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(line)  # Keep as raw JSON strings for efficiency

    total = len(records)
    print(f"Total records in dataset: {total:,}")

    # Sample if needed
    if max_records and max_records < total:
        print(f"Sampling {max_records:,} records (seed={seed})...")
        random.seed(seed)
        records = random.sample(records, max_records)
    else:
        print(f"Using all {total:,} records")
        random.seed(seed)
        random.shuffle(records)

    # Save sampled data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(record + "\n")

    print(f"Saved {len(records):,} records to {output_path}")

    # Print sample statistics
    print("\nSample statistics:")
    sample = [json.loads(r) for r in records[:100]]
    mask_keys = set()
    for rec in sample:
        masks = rec.get("masks", {})
        mask_keys.update(masks.keys())
    print(f"  Entity types found (sample of 100): {sorted(mask_keys)}")
    text_lengths = [len(rec.get("generated_sample_pii", rec.get("text", ""))) for rec in sample]
    if text_lengths:
        print(f"  Avg text length (sample): {sum(text_lengths) / len(text_lengths):.0f} chars")
        print(f"  Min/Max text length (sample): {min(text_lengths)} / {max(text_lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PII dataset for training")
    parser.add_argument("--input", required=True,
                        help="Path to output_normalized.jsonl (full dataset)")
    parser.add_argument("--output", default="data/pii_dataset.jsonl",
                        help="Path for sampled output (default: data/pii_dataset.jsonl)")
    parser.add_argument("--max-records", type=int, default=50000,
                        help="Number of records to sample (default: 50000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        max_records=args.max_records,
        seed=args.seed,
    )
