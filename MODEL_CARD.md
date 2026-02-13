---
license: apache-2.0
language:
- pt
library_name: transformers
tags:
- pii
- lgpd
- ner
- de-identification
- brazilian-portuguese
- token-classification
- privacy
datasets:
- custom
base_model: allenai/longformer-base-4096
metrics:
- f1
- precision
- recall
pipeline_tag: token-classification
---

# pii-longformer-br: Brazilian Portuguese PII Detection

PII detection and removal for Brazilian Portuguese text using a fine-tuned Longformer model with BILOU tagging.

## Model Description

This model detects 22 types of Personally Identifiable Information in Portuguese text, including Brazilian-specific document types (CPF, RG, PIS) and LGPD-sensitive categories (medical data, sexual data, race, religion, political opinion).

Detected PII is replaced with realistic surrogate data (for structured types) or redaction markers (for sensitive types), preserving text readability.

## Entity Types

### Structured PII (16 types)

| Category | Types |
|----------|-------|
| Names | FIRST_NAME, MIDDLE_NAME, LAST_NAME |
| Documents | CPF, RG, PIS, CREDIT_CARD |
| Contact | PHONE_NUMBER, EMAIL |
| Address | STREET_ADDRESS, BUILDING_NUMBER, NEIGHBORHOOD, CITY, STATE, CEP |
| Dates | DATE_OF_BIRTH |

### Sensitive Data (6 types)

MEDICAL_DATA, SEXUAL_DATA, RACE_OR_ETHNICITY, RELIGIOUS_CONVICTION, POLITICAL_OPINION, ORGANIZATION_AFFILIATION

Total: 89 labels (22 entity types x 4 BILOU tags + O)

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

text = "Maria Silva, CPF 123.456.789-00, mora na Rua Augusta, 1234, Sao Paulo - SP."

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)[0]

id2label = model.config.id2label
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for token, pred in zip(tokens, predictions):
    label = id2label[pred.item()]
    if label != "O":
        print(f"{token}: {label}")
```

## Training Details

- **Base model**: allenai/longformer-base-4096 (configurable; multilingual variant recommended for Portuguese)
- **Training data**: Brazilian Portuguese PII dataset (~50K records sampled from 650K)
- **Tagging scheme**: BILOU
- **Max sequence length**: 4,096 tokens

## Limitations

1. **Base model is English-only** — For best results, use `markussagen/xlm-roberta-longformer-base-4096` as the base
2. **Trained on synthetic data** — Real-world performance may vary
3. **Brazilian focus** — Optimized for Brazilian Portuguese documents and ID formats

## License

Apache 2.0
