# labels.py - PII entity types for Brazilian Portuguese de-identification

# Configuration flag: set to False for structured-only PII (16 types)
INCLUDE_SENSITIVE = True

# 16 Structured PII types (always included)
STRUCTURED_ENTITY_TYPES = [
    "FIRST_NAME",
    "MIDDLE_NAME",
    "LAST_NAME",
    "DATE_OF_BIRTH",
    "PHONE_NUMBER",
    "EMAIL",
    "CPF",
    "RG",
    "PIS",
    "CEP",
    "STREET_ADDRESS",
    "BUILDING_NUMBER",
    "NEIGHBORHOOD",
    "CITY",
    "STATE",
    "CREDIT_CARD",
]

# 6 Sensitive types (optional via INCLUDE_SENSITIVE flag)
SENSITIVE_ENTITY_TYPES = [
    "MEDICAL_DATA",
    "SEXUAL_DATA",
    "RACE_OR_ETHNICITY",
    "RELIGIOUS_CONVICTION",
    "POLITICAL_OPINION",
    "ORGANIZATION_AFFILIATION",
]

# BILOU tagging scheme
BILOU_TAGS = ["B", "I", "L", "U"]

# Map dataset labels → internal entity names
DATASET_TO_ENTITY = {
    "name": "FIRST_NAME",
    "middle_name": "MIDDLE_NAME",
    "surnames": "LAST_NAME",
    "birthdate": "DATE_OF_BIRTH",
    "phone": "PHONE_NUMBER",
    "email": "EMAIL",
    "cpf": "CPF",
    "rg": "RG",
    "pis": "PIS",
    "cep": "CEP",
    "street": "STREET_ADDRESS",
    "building_number": "BUILDING_NUMBER",
    "neighborhood": "NEIGHBORHOOD",
    "city_name": "CITY",
    "state": "STATE",
    "state_abbr": "STATE",
    "creditcard": "CREDIT_CARD",
    # Sensitive
    "medical_data": "MEDICAL_DATA",
    "sexual_data": "SEXUAL_DATA",
    "race_or_ethnicity": "RACE_OR_ETHNICITY",
    "religious_conviction": "RELIGIOUS_CONVICTION",
    "political_opinion": "POLITICAL_OPINION",
    "organization_affiliation": "ORGANIZATION_AFFILIATION",
}


def build_label_maps(include_sensitive=True):
    """Build BILOU label mappings.

    Args:
        include_sensitive: If True, include sensitive entity types (22 total).
                          If False, only structured PII types (16 total).

    Returns:
        (entity_types, labels, label2id, id2label)
    """
    entity_types = list(STRUCTURED_ENTITY_TYPES)
    if include_sensitive:
        entity_types.extend(SENSITIVE_ENTITY_TYPES)

    labels = ["O"]  # Outside tag first
    for entity in entity_types:
        for tag in BILOU_TAGS:
            labels.append(f"{tag}-{entity}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    return entity_types, labels, label2id, id2label


# Build default mappings based on INCLUDE_SENSITIVE flag
ENTITY_TYPES, LABELS, LABEL2ID, ID2LABEL = build_label_maps(INCLUDE_SENSITIVE)
NUM_LABELS = len(LABELS)  # 89 (full) or 65 (structured only)

# Build dataset-to-entity mapping filtered by active entity types
ACTIVE_ENTITY_SET = set(ENTITY_TYPES)
ACTIVE_DATASET_TO_ENTITY = {
    k: v for k, v in DATASET_TO_ENTITY.items() if v in ACTIVE_ENTITY_SET
}

if __name__ == "__main__":
    print(f"Include sensitive: {INCLUDE_SENSITIVE}")
    print(f"Entity types: {len(ENTITY_TYPES)}")
    print(f"Total labels: {NUM_LABELS}")
    print(f"\nEntity types: {ENTITY_TYPES}")
    print(f"\nFirst 10 labels: {LABELS[:10]}")
    print(f"Last 10 labels: {LABELS[-10:]}")
    print(f"\nDataset mappings: {len(ACTIVE_DATASET_TO_ENTITY)}")
