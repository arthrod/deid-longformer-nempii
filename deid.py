# deid.py - PII De-identification Surrogate Generator (Brazilian Portuguese)
#
# Key features:
# - Consistent replacement within documents (same PII → same fake)
# - Format preservation (phone, CPF, dates match original format)
# - Realistic surrogates using Faker pt_BR locale
# - Brazilian document generators (CPF, RG, PIS, CEP, credit card)
# - Date shifting preserves temporal relationships
# - Name normalization for cache consistency
# - Sensitive data redacted with marker text (no realistic surrogates)
# - Adjacent name combination for consistent physician/person names

from faker import Faker
from datetime import datetime, timedelta
import random
import re
from typing import Optional, Tuple

from labels import SENSITIVE_ENTITY_TYPES

# Redaction marker for sensitive data types
SENSITIVE_REDACTION_MARKER = "[DADO SENSIVEL REMOVIDO]"


def normalize_name(name: str) -> str:
    """
    Normalize a person name so that variants like
    'Dr. Maria Elizabeth Silva' and 'Maria E. Silva'
    map to the same cache key.
    """
    n = name.lower()
    # Remove common Portuguese titles/credentials
    n = re.sub(
        r'\b(dr\.?|dra\.?|prof\.?|profa\.?|sr\.?|sra\.?|eng\.?|adv\.?'
        r'|me\.?|ms\.?|phd|jr\.?|neto|filho|filha|sobrinho|sobrinha)\b',
        '', n
    )
    # Remove non-letter characters except spaces
    n = re.sub(r'[^a-záàãâéêíóôõúç\s]', ' ', n)
    # Collapse multiple spaces
    n = re.sub(r'\s+', ' ', n).strip()
    parts = n.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}"
    return n.strip()


def _combine_adjacent_name_entities(entities: list, text: str) -> list:
    """
    Combine adjacent FIRST_NAME/MIDDLE_NAME/LAST_NAME entities into single NAME entities.
    This ensures that split names get the same cache key.
    """
    name_types = {"FIRST_NAME", "MIDDLE_NAME", "LAST_NAME", "NAME"}

    sorted_ents = sorted(entities, key=lambda x: x["start"])

    combined = []
    i = 0
    while i < len(sorted_ents):
        curr = sorted_ents[i]

        if curr["type"] not in name_types:
            combined.append(curr.copy())
            i += 1
            continue

        group_start = curr["start"]
        group_end = curr["end"]
        j = i + 1

        while j < len(sorted_ents):
            next_ent = sorted_ents[j]
            if next_ent["type"] not in name_types:
                break

            gap = next_ent["start"] - group_end
            if gap > 15:
                break

            gap_text = text[group_end:next_ent["start"]]
            if gap_text and not re.match(
                r'^[\s.,\-\']+[A-ZÀ-Ú]?\.?[\s.,\-\']*$|^[\s.,\-\']*$', gap_text
            ):
                break

            group_end = next_ent["end"]
            j += 1

        if j > i + 1:
            combined_text = text[group_start:group_end]
            combined.append({
                "type": "NAME",
                "start": group_start,
                "end": group_end,
                "text": combined_text,
            })
            i = j
        else:
            combined.append(curr.copy())
            i += 1

    return combined


class PIIDeidentifier:
    def __init__(self, seed: Optional[int] = None):
        self.fake = Faker('pt_BR')
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self._cache = {}
        self._date_shift = None
        self._current_location = None
        self._full_name_cache = {}

    def reset_cache(self):
        """Call between documents to reset consistency caches."""
        self._cache = {}
        self._date_shift = random.randint(-365, -30)
        self._current_location = None
        self._full_name_cache = {}

    def _get_cached(self, original: str, entity_type: str, generator_fn):
        """Ensure consistent replacement within a document."""
        if entity_type in ("FIRST_NAME", "MIDDLE_NAME", "LAST_NAME", "NAME"):
            norm = normalize_name(original)
            key = f"NAME:{norm}"
        else:
            key = f"{entity_type}:{original.strip()}"

        if key not in self._cache:
            self._cache[key] = generator_fn()
        return self._cache[key]

    def _preserve_case(self, original: str, replacement: str) -> str:
        """Match the case pattern of the original text."""
        if original.isupper():
            return replacement.upper()
        elif original.islower():
            return replacement.lower()
        elif original.istitle():
            return replacement.title()
        return replacement

    def replace(self, text: str, entity_type: str) -> str:
        """Replace detected PII with realistic surrogate data."""

        # === SENSITIVE DATA — redact, don't generate surrogates ===
        if entity_type in SENSITIVE_ENTITY_TYPES:
            return SENSITIVE_REDACTION_MARKER

        # === NAMES ===
        if entity_type == "FIRST_NAME":
            replacement = self._get_cached(text, entity_type, self.fake.first_name)
            return self._preserve_case(text, replacement)

        elif entity_type == "MIDDLE_NAME":
            replacement = self._get_cached(text, entity_type, self.fake.first_name)
            return self._preserve_case(text, replacement)

        elif entity_type == "LAST_NAME":
            replacement = self._get_cached(text, entity_type, self.fake.last_name)
            return self._preserve_case(text, replacement)

        elif entity_type == "NAME":
            replacement = self._get_cached(text, entity_type, self.fake.name)
            return self._preserve_case(text, replacement)

        # === DATE OF BIRTH ===
        elif entity_type == "DATE_OF_BIRTH":
            return self._generate_dob(text)

        # === BRAZILIAN DOCUMENTS ===
        elif entity_type == "CPF":
            return self._get_cached(text, entity_type, self._generate_cpf)

        elif entity_type == "RG":
            return self._get_cached(text, entity_type, self._generate_rg)

        elif entity_type == "PIS":
            return self._get_cached(text, entity_type, self._generate_pis)

        elif entity_type == "CREDIT_CARD":
            return self._get_cached(text, entity_type, self._generate_credit_card)

        # === CONTACT INFO ===
        elif entity_type == "PHONE_NUMBER":
            return self._generate_brazilian_phone(text)

        elif entity_type == "EMAIL":
            return self._get_cached(text, entity_type, self._generate_email)

        # === ADDRESSES ===
        elif entity_type == "STREET_ADDRESS":
            return self._get_cached(text, entity_type, self._generate_street)

        elif entity_type == "BUILDING_NUMBER":
            return str(random.randint(1, 9999))

        elif entity_type == "NEIGHBORHOOD":
            return self._get_cached(
                text, entity_type,
                lambda: self.fake.bairro() if hasattr(self.fake, 'bairro') else self.fake.city_suffix() + " " + self.fake.last_name()
            )

        elif entity_type == "CITY":
            return self._get_cached(text, entity_type, self._generate_city)

        elif entity_type == "STATE":
            return self._get_location()[1]

        elif entity_type == "CEP":
            return self._get_cached(text, entity_type, self._generate_cep)

        # === FALLBACK ===
        else:
            return self._generate_generic_id(text)

    # === BRAZILIAN DOCUMENT GENERATORS ===

    def _generate_cpf(self) -> str:
        """Generate fake CPF in format XXX.XXX.XXX-XX."""
        digits = [random.randint(0, 9) for _ in range(9)]
        # Calculate first check digit
        s = sum(d * w for d, w in zip(digits, range(10, 1, -1)))
        d1 = 11 - (s % 11)
        d1 = 0 if d1 >= 10 else d1
        digits.append(d1)
        # Calculate second check digit
        s = sum(d * w for d, w in zip(digits, range(11, 1, -1)))
        d2 = 11 - (s % 11)
        d2 = 0 if d2 >= 10 else d2
        digits.append(d2)
        return f"{digits[0]}{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}.{digits[6]}{digits[7]}{digits[8]}-{digits[9]}{digits[10]}"

    def _generate_rg(self) -> str:
        """Generate fake RG in format XX.XXX.XXX-X."""
        d = [random.randint(0, 9) for _ in range(8)]
        check = random.randint(0, 9)
        return f"{d[0]}{d[1]}.{d[2]}{d[3]}{d[4]}.{d[5]}{d[6]}{d[7]}-{check}"

    def _generate_pis(self) -> str:
        """Generate fake PIS/PASEP in format XXX.XXXX.XXX-X."""
        d = [random.randint(0, 9) for _ in range(10)]
        check = random.randint(0, 9)
        return f"{d[0]}{d[1]}{d[2]}.{d[3]}{d[4]}{d[5]}{d[6]}.{d[7]}{d[8]}{d[9]}-{check}"

    def _generate_cep(self) -> str:
        """Generate fake CEP in format XXXXX-XXX."""
        return f"{random.randint(10000, 99999)}-{random.randint(100, 999)}"

    def _generate_credit_card(self) -> str:
        """Generate fake credit card in format XXXX XXXX XXXX XXXX."""
        groups = [f"{random.randint(1000, 9999)}" for _ in range(4)]
        return " ".join(groups)

    def _generate_brazilian_phone(self, original: str) -> str:
        """Generate fake Brazilian phone number, preserving format."""
        original = original.strip()
        ddd = random.randint(11, 99)
        number = random.randint(90000, 99999) * 10000 + random.randint(0, 9999)
        first = number // 10000
        last = number % 10000

        # Detect format from original
        if re.match(r'\(\d{2}\)\s*\d{4,5}[-.]?\d{4}', original):
            return f"({ddd:02d}) {first}-{last:04d}"
        elif re.match(r'\d{2}\s*\d{4,5}[-.]?\d{4}', original):
            return f"{ddd:02d} {first}-{last:04d}"
        elif re.match(r'\+55', original):
            return f"+55 ({ddd:02d}) {first}-{last:04d}"
        else:
            return f"({ddd:02d}) {first}-{last:04d}"

    # === DATE HANDLING ===

    def _generate_dob(self, text: str) -> str:
        """Generate realistic DOB preserving approximate age (±2 years)."""
        parsed = self._parse_date(text)
        if parsed:
            dt, fmt = parsed
            year_jitter = random.randint(-2, 2)
            new_month = random.randint(1, 12)
            new_day = random.randint(1, 28)
            try:
                shifted = dt.replace(year=dt.year + year_jitter, month=new_month, day=new_day)
                return shifted.strftime(fmt)
            except ValueError:
                shifted = dt.replace(year=dt.year + year_jitter, month=new_month, day=15)
                return shifted.strftime(fmt)
        return self.fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%d/%m/%Y")

    def _shift_date(self, text: str) -> str:
        """Shift date by consistent offset, preserving original format."""
        if self._date_shift is None:
            self._date_shift = random.randint(-365, -30)

        text = text.strip()
        parsed = self._parse_date(text)
        if parsed:
            dt, fmt = parsed
            shifted = dt + timedelta(days=self._date_shift)
            return shifted.strftime(fmt)

        reference_date = datetime(2025, 1, 15)
        shifted = reference_date + timedelta(days=self._date_shift)
        return shifted.strftime("%d/%m/%Y")

    def _parse_date(self, text: str) -> Optional[Tuple[datetime, str]]:
        """Parse date and return (datetime, format_string)."""
        formats = [
            # Brazilian formats (DD/MM/YYYY)
            ("%d/%m/%Y", "%d/%m/%Y"),
            ("%d-%m-%Y", "%d-%m-%Y"),
            ("%d/%m/%y", "%d/%m/%y"),
            ("%d-%m-%y", "%d-%m-%y"),
            # ISO formats
            ("%Y-%m-%d", "%Y-%m-%d"),
            # Written formats (Portuguese)
            ("%d de %B de %Y", "%d de %B de %Y"),
            # US formats (less common but possible in data)
            ("%m/%d/%Y", "%m/%d/%Y"),
        ]

        for parse_fmt, out_fmt in formats:
            try:
                dt = datetime.strptime(text.strip(), parse_fmt)
                if 1900 <= dt.year <= 2100:
                    return (dt, out_fmt)
            except ValueError:
                continue

        # Try regex for dates like "15/03/1990" with flexible separators
        match = re.match(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', text)
        if match:
            d, m, y = match.groups()
            try:
                if len(y) == 2:
                    y = "20" + y if int(y) < 50 else "19" + y
                dt = datetime(int(y), int(m), int(d))
                sep = "/" if "/" in text else "-"
                if len(match.group(3)) == 2:
                    fmt = f"%d{sep}%m{sep}%y"
                else:
                    fmt = f"%d{sep}%m{sep}%Y"
                return (dt, fmt)
            except ValueError:
                pass

        return None

    # === CONTACT GENERATION ===

    def _generate_email(self) -> str:
        """Generate realistic email."""
        return self.fake.email()

    # === ADDRESS GENERATION ===

    def _get_location(self) -> Tuple[str, str, str]:
        """Get consistent city, state, CEP for this document."""
        if self._current_location is None:
            city = self.fake.city()
            state = self.fake.state_abbr()
            cep = self._generate_cep()
            self._current_location = (city, state, cep)
        return self._current_location

    def _generate_street(self) -> str:
        """Generate realistic street address."""
        return self.fake.street_name()

    def _generate_city(self) -> str:
        """Generate city (consistent within document)."""
        return self._get_location()[0]

    # === GENERIC FALLBACK ===

    def _generate_generic_id(self, original: str) -> str:
        """Generate fake ID preserving approximate length."""
        original = original.strip()
        alphanum = re.findall(r'[A-Za-z0-9]', original)
        length = len(alphanum) if alphanum else 8
        chars = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=max(4, length)))
        return chars


# === Utility for batch processing ===

def deidentify_text(text: str, entities: list, seed: int = None) -> str:
    """
    Replace entities in text with surrogate data.

    Args:
        text: Original text
        entities: List of dicts with 'type', 'start', 'end', 'text' keys
        seed: Random seed for reproducibility

    Returns:
        De-identified text
    """
    deid = PIIDeidentifier(seed=seed)
    deid.reset_cache()

    # Pre-process: Combine adjacent name entities for cache consistency
    entities = _combine_adjacent_name_entities(entities, text)

    # Sort by start position (reverse) for safe replacement
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

    result = text
    for entity in sorted_entities:
        entity_type = entity["type"]
        replacement = deid.replace(entity["text"], entity_type)
        result = result[:entity["start"]] + replacement + result[entity["end"]:]

    return result
