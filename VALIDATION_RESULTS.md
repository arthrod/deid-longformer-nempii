# Validation Results

Validation results for the PII detection model on Brazilian Portuguese text.

## Results Summary

| Dataset | F1 | Precision | Recall | Notes | Date |
|---------|-----|-----------|--------|-------|------|
| PII Dataset (held-out) | -- | -- | -- | *Pending first training run* | -- |

---

## How to Submit Results

1. Train the model using `train.py`
2. Record validation metrics from the training output
3. Add results below in this format:

```markdown
### [Dataset / Split Description]

**Date:** [YYYY-MM-DD]
**Model:** [base model used]
**Training Records:** [number]
**Validation Records:** [number]

#### Overall Metrics
| Metric | Value |
|--------|-------|
| F1 | XX.XX% |
| Precision | XX.XX% |
| Recall | XX.XX% |

#### Per-Entity Breakdown (Optional)
| Entity Type | F1 | Support |
|-------------|-----|---------|
| FIRST_NAME | XX.XX% | XXX |
| CPF | XX.XX% | XXX |
| ... | ... | ... |

#### Notes
[Any observations about failure modes, edge cases, etc.]
```
