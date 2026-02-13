# run_baseline.py - End-to-end training + HuggingFace upload pipeline

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from prepare_data import prepare_dataset
from preprocess import preprocess_dataset
from train import train as train_model


SAMPLED_PATH = "data/pii_dataset.jsonl"
PROCESSED_PATH = "data/processed_bilou.jsonl"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best_model")


def upload_to_hub(repo_id, model_dir, model_card_path, token):
    """Push trained model, tokenizer, and model card to HuggingFace Hub."""
    from huggingface_hub import HfApi
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_dir = Path(model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    print(f"Loading model from {model_dir}...")
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Pushing model to {repo_id} (private)...")
    model.push_to_hub(repo_id, private=True, token=token)
    tokenizer.push_to_hub(repo_id, private=True, token=token)

    # Upload MODEL_CARD.md as the repo README
    model_card = Path(model_card_path)
    if model_card.exists():
        print(f"Uploading {model_card} as README.md...")
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
    else:
        print(f"Warning: {model_card} not found, skipping README upload")

    print(f"Done! Model available at https://huggingface.co/{repo_id}")


def resolve_hf_token(cli_token):
    """Resolve HuggingFace token from CLI arg, env var, or cached login."""
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    # Fall back to cached login (huggingface-cli login)
    try:
        from huggingface_hub import get_token
        cached = get_token()
        if cached:
            return cached
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end PII model training + HuggingFace upload"
    )
    parser.add_argument("--input", required=True,
                        help="Path to output_normalized.jsonl")
    parser.add_argument("--repo-id", required=True,
                        help='HuggingFace repo name, e.g. "username/pii-longformer-br"')
    parser.add_argument("--model-name", default="allenai/longformer-base-4096",
                        help="Base model (default: allenai/longformer-base-4096)")
    parser.add_argument("--max-records", type=int, default=50000,
                        help="Records to sample (default: 50000)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-sensitive", action="store_true",
                        help="Exclude sensitive entity types")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or uses HF_TOKEN env var / cached login)")
    args = parser.parse_args()

    include_sensitive = not args.no_sensitive

    # Resolve HF token early so we fail fast
    token = resolve_hf_token(args.hf_token)
    if not token:
        print("Error: No HuggingFace token found. Provide --hf-token, "
              "set HF_TOKEN env var, or run `huggingface-cli login`.")
        sys.exit(1)

    print("=" * 60)
    print("PII Longformer — End-to-end Pipeline")
    print("=" * 60)
    print(f"  Input:        {args.input}")
    print(f"  Repo:         {args.repo_id}")
    print(f"  Base model:   {args.model_name}")
    print(f"  Max records:  {args.max_records:,}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Seed:         {args.seed}")
    print(f"  Sensitive:    {include_sensitive}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1/4: Sample data
    # ------------------------------------------------------------------
    print(f"\nStep 1/4: Sampling {args.max_records:,} records...")
    prepare_dataset(
        input_path=args.input,
        output_path=SAMPLED_PATH,
        max_records=args.max_records,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Step 2/4: Preprocess to BILOU
    # ------------------------------------------------------------------
    print(f"\nStep 2/4: Preprocessing to BILOU labels...")
    preprocess_dataset(
        input_path=SAMPLED_PATH,
        output_path=PROCESSED_PATH,
        model_name=args.model_name,
        include_sensitive=include_sensitive,
    )

    # ------------------------------------------------------------------
    # Step 3/4: Train
    # ------------------------------------------------------------------
    print(f"\nStep 3/4: Training for {args.epochs} epochs...")
    train_args = SimpleNamespace(
        data=PROCESSED_PATH,
        output=CHECKPOINT_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model_name,
    )
    train_model(train_args)

    # ------------------------------------------------------------------
    # Step 4/4: Upload to HuggingFace
    # ------------------------------------------------------------------
    print(f"\nStep 4/4: Uploading to HuggingFace ({args.repo_id})...")
    upload_to_hub(
        repo_id=args.repo_id,
        model_dir=BEST_MODEL_DIR,
        model_card_path="MODEL_CARD.md",
        token=token,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
