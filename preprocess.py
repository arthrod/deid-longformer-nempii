# preprocess.py - Convert PII dataset (masks format) to BILOU token labels

import json
import argparse
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from labels import build_label_maps

DEFAULT_MODEL = "allenai/longformer-base-4096"
MAX_LENGTH = 4096


def load_data(path, max_records=None):
    """Load JSONL data, optionally limiting to max_records."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
            if max_records and len(records) >= max_records:
                break
    return records


def extract_spans_from_masks(text, masks, entity_mapping):
    """
    Extract character-level spans by finding mask values in text.

    The dataset provides masks as {entity_type: value} pairs.
    We find ALL occurrences of each value in the text.

    Args:
        text: The generated_sample_pii text
        masks: Dict of {entity_type: value} from the dataset
        entity_mapping: Map from dataset labels to internal entity names

    Returns:
        List of span dicts with 'start', 'end', 'label' keys
    """
    spans = []

    for mask_key, mask_value in masks.items():
        # Skip if this entity type is not in our mapping
        if mask_key not in entity_mapping:
            continue

        entity_type = entity_mapping[mask_key]

        # Skip empty values
        if not mask_value or not str(mask_value).strip():
            continue

        value_str = str(mask_value).strip()

        # Find ALL occurrences of this value in the text
        start = 0
        while True:
            idx = text.find(value_str, start)
            if idx == -1:
                break
            spans.append({
                "start": idx,
                "end": idx + len(value_str),
                "label": entity_type,
            })
            start = idx + 1  # Move past this occurrence

    return spans


def resolve_overlaps(spans):
    """
    Resolve overlapping spans. Longer span wins; on ties, earlier span wins.

    Args:
        spans: List of span dicts with 'start', 'end', 'label'

    Returns:
        Non-overlapping list of spans
    """
    if not spans:
        return []

    # Sort by length (descending), then by start position (ascending)
    sorted_spans = sorted(spans, key=lambda s: (-(s["end"] - s["start"]), s["start"]))

    resolved = []
    occupied = []  # List of (start, end) tuples for accepted spans

    for span in sorted_spans:
        s_start, s_end = span["start"], span["end"]

        # Check overlap with already accepted spans
        overlap = False
        for occ_start, occ_end in occupied:
            if s_start < occ_end and s_end > occ_start:
                overlap = True
                break

        if not overlap:
            resolved.append(span)
            occupied.append((s_start, s_end))

    # Sort by start position for output
    resolved.sort(key=lambda s: s["start"])
    return resolved


def spans_to_bilou(text, spans, tokenizer, label2id):
    """
    Convert character-level spans to token-level BILOU labels.

    Returns:
        dict with input_ids, attention_mask, labels
    """
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )

    offset_mapping = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Initialize all labels as "O"
    labels = [label2id["O"]] * len(input_ids)

    # For each span, find which tokens it covers
    for span in spans:
        span_start = span["start"]
        span_end = span["end"]
        entity = span["label"]

        # Find tokens that overlap with this span
        span_token_indices = []
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            # Skip special tokens (offset is (0,0))
            if token_start == 0 and token_end == 0 and idx != 0:
                continue
            # Check for overlap
            if token_end > span_start and token_start < span_end:
                span_token_indices.append(idx)

        if len(span_token_indices) == 0:
            continue

        # Check that BILOU labels exist for this entity
        u_label = f"U-{entity}"
        if u_label not in label2id:
            continue

        if len(span_token_indices) == 1:
            idx = span_token_indices[0]
            labels[idx] = label2id[f"U-{entity}"]
        else:
            for i, idx in enumerate(span_token_indices):
                if i == 0:
                    labels[idx] = label2id[f"B-{entity}"]
                elif i == len(span_token_indices) - 1:
                    labels[idx] = label2id[f"L-{entity}"]
                else:
                    labels[idx] = label2id[f"I-{entity}"]

    # Set labels for special tokens to -100 (ignored in loss)
    for idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_start == 0 and token_end == 0:
            labels[idx] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess_dataset(input_path, output_path, model_name, max_records=None,
                       include_sensitive=True):
    """Process full dataset."""
    # Build label maps for the requested configuration
    entity_types, all_labels, label2id, id2label = build_label_maps(include_sensitive)
    active_entity_set = set(entity_types)

    # Filter dataset-to-entity mapping
    from labels import DATASET_TO_ENTITY
    entity_mapping = {k: v for k, v in DATASET_TO_ENTITY.items() if v in active_entity_set}

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading data from {input_path}")
    records = load_data(input_path, max_records)
    print(f"Loaded {len(records)} records")
    print(f"Entity types: {len(entity_types)} ({len(all_labels)} labels)")

    processed = []
    entity_counts = Counter()
    skipped = 0
    no_spans = 0

    for i, record in enumerate(records):
        # Support both field names from the dataset
        text = record.get("generated_sample_pii") or record.get("text", "")
        masks = record.get("masks", {})

        if not text:
            skipped += 1
            continue

        # Extract spans from masks
        raw_spans = extract_spans_from_masks(text, masks, entity_mapping)

        # Resolve overlaps
        spans = resolve_overlaps(raw_spans)

        if not spans:
            no_spans += 1

        # Count entities for stats
        for span in spans:
            entity_counts[span["label"]] += 1

        try:
            result = spans_to_bilou(text, spans, tokenizer, label2id)
            # Preserve record ID if available
            if "id" in record:
                result["record_id"] = record["id"]
            elif "uid" in record:
                result["record_id"] = record["uid"]
            processed.append(result)
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            skipped += 1
            continue

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(records)}")

    print(f"\nProcessed: {len(processed)}, Skipped: {skipped}, No spans: {no_spans}")

    print(f"\nEntity counts:")
    for entity, count in entity_counts.most_common():
        print(f"  {entity}: {count}")

    # Save processed data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in processed:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(processed)} records to {output_path}")
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PII dataset to BILOU token labels")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", default="data/processed_bilou.jsonl", help="Path for output JSONL")
    parser.add_argument("--max-records", type=int, default=50000, help="Maximum records to process")
    parser.add_argument("--include-sensitive", action="store_true", default=True,
                        help="Include sensitive entity types")
    parser.add_argument("--no-sensitive", action="store_true",
                        help="Exclude sensitive entity types")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                        help=f"Model name for tokenizer (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    include_sensitive = not args.no_sensitive

    preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        max_records=args.max_records,
        include_sensitive=include_sensitive,
    )
