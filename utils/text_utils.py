import re
import unicodedata

def normalize_text(text):
    """
    Normalize text for better matching:
    - Lowercase
    - Remove punctuation
    - Normalize Romanian diacritics
    - Strip whitespace
    """
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Handle Romanian-specific diacritics (Legacy/Common variations)
    # ş -> ș, ţ -> ț
    text = text.replace('ş', 'ș').replace('ţ', 'ț')

    # 3. Optional: Strip all diacritics for even more aggressive matching
    # Useful as a fallback if exact match fails
    # text = ''.join(c for c in unicodedata.normalize('NFD', text)
    #                if unicodedata.category(c) != 'Mn')

    # 4. Remove punctuation (except brackets if we want to keep them for padding logic later)
    # Actually, for matching, we should remove brackets too.
    text = re.sub(r'[^\w\s]', '', text)

    # 5. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def strip_diacritics(text):
    """Aggressively remove all diacritics for broad matching"""
    nks = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nks if not unicodedata.combining(c)])

def clean_text_for_alignment(text):
    """
    Remove tags, speaker names, and extra formatting that shouldn't be matched:
    - Removes [tags in brackets]
    - Removes Speaker: prefix
    - Normalizes diacritics
    """
    if not text:
        return ""

    # 1. Remove [bracketed tags]
    text = re.sub(r'\[.*?\]', '', text)

    # 2. Remove Speaker: prefix (if it's at the start and followed by text)
    # Match strings like "Name: " or "Character Name: "
    text = re.sub(r'^[A-Za-z\s]+:\s*', '', text)

    # 3. Standard normalization
    text = normalize_text(text)

    return text
