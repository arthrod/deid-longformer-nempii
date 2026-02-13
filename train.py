# train.py

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    LongformerForTokenClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from labels import LABELS, LABEL2ID, ID2LABEL, NUM_LABELS, ENTITY_TYPES

DEFAULT_MODEL = "allenai/longformer-base-4096"

class DeidDataset(Dataset):
    def __init__(self, records):
        self.records = records
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(record["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(record["labels"], dtype=torch.long)
        }

def load_processed_data(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def compute_class_weights(records):
    """Compute class weights for imbalanced labels."""
    label_counts = Counter()
    total = 0
    
    for record in records:
        for label in record["labels"]:
            if label != -100:  # Skip ignored tokens
                label_counts[label] += 1
                total += 1
    
    # Compute weights (inverse frequency, capped)
    weights = torch.ones(NUM_LABELS)
    for label_id, count in label_counts.items():
        if count > 0:
            # Inverse frequency with smoothing
            weights[label_id] = min(total / (count * NUM_LABELS), 10.0)
    
    return weights

def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            
            # Flatten and filter out ignored tokens
            for pred_seq, label_seq in zip(preds, labels):
                for pred, label in zip(pred_seq, label_seq):
                    if label != -100:
                        all_preds.append(pred.item())
                        all_labels.append(label.item())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    
    # Entity-level metrics (non-O labels)
    entity_mask = all_labels != LABEL2ID["O"]
    if entity_mask.sum() > 0:
        entity_preds = all_preds[entity_mask]
        entity_labels = all_labels[entity_mask]
        entity_accuracy = (entity_preds == entity_labels).mean()
        
        # Precision, Recall, F1 for entity detection (binary: entity vs O)
        pred_entity = all_preds != LABEL2ID["O"]
        true_entity = all_labels != LABEL2ID["O"]
        
        tp = (pred_entity & true_entity).sum()
        fp = (pred_entity & ~true_entity).sum()
        fn = (~pred_entity & true_entity).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        entity_accuracy = 0
        precision = recall = f1 = 0
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "entity_accuracy": entity_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\nLoading data from {args.data}")
    records = load_processed_data(args.data)
    print(f"Total records: {len(records)}")
    
    # Train/val split
    train_records, val_records = train_test_split(
        records, test_size=0.2, random_state=42
    )
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")
    
    # Create datasets
    train_dataset = DeidDataset(train_records)
    val_dataset = DeidDataset(val_records)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Compute class weights
    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_records).to(device)
    
    # Load model
    model_name = args.model_name
    print(f"\nLoading model: {model_name}")
    model = LongformerForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"{'='*60}\n")
    
    best_f1 = 0
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Compute loss with class weights
            logits = outputs.logits.view(-1, NUM_LABELS)
            labels_flat = labels.view(-1)
            loss = loss_fn(logits, labels_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Entity Accuracy: {val_metrics['entity_accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            print(f"  New best F1! Saving model...")
            model.save_pretrained(output_dir / "best_model")
            
            # Save tokenizer too for easy loading
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(output_dir / "best_model")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir / 'best_model'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed_bilou.jsonl")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model_name", default=DEFAULT_MODEL,
                        help=f"Base model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    
    train(args)