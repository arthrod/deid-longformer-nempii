# pii-longformer-br

**General-purpose PII removal for Brazilian Portuguese text using Longformer.**

A fine-tuned Longformer model for detecting and replacing Personally Identifiable Information (PII) in Portuguese text, supporting 22 entity types including Brazilian documents (CPF, RG, PIS) and LGPD-sensitive categories.

## Why This Exists

Brazilian data protection law (LGPD - Lei Geral de Protecao de Dados) requires organizations to protect personal data. This model automates PII detection and generates **realistic surrogate data** instead of simple redaction, preserving text readability for downstream NLP tasks.

## Entity Types

### Structured PII (16 types, always active)

| Entity | Description | Example |
|--------|-------------|---------|
| FIRST_NAME | First name | Maria |
| MIDDLE_NAME | Middle name | Elizabeth |
| LAST_NAME | Last name / surnames | Silva |
| DATE_OF_BIRTH | Date of birth | 15/03/1990 |
| PHONE_NUMBER | Phone number | (11) 98765-4321 |
| EMAIL | Email address | maria@example.com |
| CPF | Cadastro de Pessoa Fisica | 123.456.789-00 |
| RG | Registro Geral | 12.345.678-9 |
| PIS | PIS/PASEP number | 123.4567.890-1 |
| CEP | Postal code (CEP) | 01310-100 |
| STREET_ADDRESS | Street name | Rua Augusta |
| BUILDING_NUMBER | Building number | 1234 |
| NEIGHBORHOOD | Neighborhood | Consolacao |
| CITY | City name | Sao Paulo |
| STATE | State / state abbreviation | SP |
| CREDIT_CARD | Credit card number | 4111 1111 1111 1111 |

### Sensitive Data (6 types, optional via config)

| Entity | Description | Replacement |
|--------|-------------|-------------|
| MEDICAL_DATA | Medical/health information | [DADO SENSIVEL REMOVIDO] |
| SEXUAL_DATA | Sexual orientation/behavior | [DADO SENSIVEL REMOVIDO] |
| RACE_OR_ETHNICITY | Race or ethnicity | [DADO SENSIVEL REMOVIDO] |
| RELIGIOUS_CONVICTION | Religious beliefs | [DADO SENSIVEL REMOVIDO] |
| POLITICAL_OPINION | Political opinions | [DADO SENSIVEL REMOVIDO] |
| ORGANIZATION_AFFILIATION | Organization membership | [DADO SENSIVEL REMOVIDO] |

Sensitive data types are redacted with a marker rather than replaced with surrogates.

## Model Architecture

```
Base Model:     allenai/longformer-base-4096 (configurable)
Max Length:     4,096 tokens
Task:           Token Classification (NER)
Tagging:        BILOU scheme
Classes:        89 (22 entity types x 4 BILOU tags + O)
```

**Note:** `allenai/longformer-base-4096` is English-only. For better Portuguese performance, set the model to a multilingual variant:

```bash
# Recommended for Portuguese:
python train.py --model_name markussagen/xlm-roberta-longformer-base-4096
```

## Key Features

### Brazilian Document Generation
Generates valid-format surrogate CPFs, RGs, PIS numbers, CEPs, and Brazilian phone numbers.

### Consistent Replacement
Same PII within a document always maps to the same surrogate:
- "Maria Silva" on line 1 and line 50 both become "Ana Oliveira"

### Format Preservation
Phone numbers, documents, and dates maintain their original format:
- `(11) 98765-4321` -> `(21) 93456-7890`
- `123.456.789-00` -> `987.654.321-00`
- `15/03/1990` -> `22/07/1988`

### Name Normalization
Variants of the same person map to the same fake name:
- "Dr. Maria Silva" -> "Dra. Ana Oliveira"
- "Maria S." -> "Ana O."

### LGPD Sensitive Data
Sensitive categories (medical, sexual, racial, religious, political) are redacted rather than replaced with surrogates.

## Installation

```bash
git clone <repository-url>
cd pii-longformer-br
pip install -r requirements.txt
```

## Data Preparation

```bash
# Sample 50K records from the full dataset
python prepare_data.py \
  --input /path/to/output_normalized.jsonl \
  --output data/pii_dataset.jsonl \
  --max-records 50000
```

## Preprocessing

```bash
# Convert to BILOU token labels
python preprocess.py \
  --input data/pii_dataset.jsonl \
  --output data/processed_bilou.jsonl \
  --max-records 50000 \
  --model-name allenai/longformer-base-4096
```

## Training

```bash
python train.py \
  --data data/processed_bilou.jsonl \
  --output checkpoints \
  --epochs 10 \
  --batch_size 4 \
  --lr 5e-5 \
  --model_name allenai/longformer-base-4096
```

## Quick Start (Python API)

```python
from deid import deidentify_text

text = """
Nome: Maria Elizabeth Silva
CPF: 123.456.789-00
Telefone: (11) 98765-4321
Endereco: Rua Augusta, 1234 - Consolacao, Sao Paulo - SP, CEP 01310-100
"""

entities = [
    {"type": "FIRST_NAME", "start": 6, "end": 11, "text": "Maria"},
    {"type": "MIDDLE_NAME", "start": 12, "end": 21, "text": "Elizabeth"},
    {"type": "LAST_NAME", "start": 22, "end": 27, "text": "Silva"},
    {"type": "CPF", "start": 33, "end": 47, "text": "123.456.789-00"},
    # ... more entities
]

result = deidentify_text(text, entities)
print(result)
```

## FastAPI Service

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST http://localhost:8001/deid/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Nome: Maria Silva, CPF: 123.456.789-00"}'
```

## Configuration

### Sensitive Entity Toggle

In `labels.py`, set `INCLUDE_SENSITIVE = False` to use only the 16 structured PII types (65 labels instead of 89).

### Model Selection

The base model is configurable via CLI args in both `preprocess.py` and `train.py`:

```bash
# English-only (default, faster)
--model-name allenai/longformer-base-4096

# Multilingual (recommended for Portuguese)
--model-name markussagen/xlm-roberta-longformer-base-4096
```

## License

Apache 2.0

## Training Data

Brazilian Portuguese PII dataset (`output_normalized.jsonl`, ~650K records).

## Acknowledgments

- Base architecture: [allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096)
- Multilingual variant: [markussagen/xlm-roberta-longformer-base-4096](https://huggingface.co/markussagen/xlm-roberta-longformer-base-4096)
- Original clinical de-identification system: [obi/deid_bert_i2b2](https://huggingface.co/obi/deid_bert_i2b2)
- Surrogate generation approach inspired by [deid-LONGFORMER-NemPII](https://github.com/Hrygt/deid-longformer-nempii)
