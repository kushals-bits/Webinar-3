# -*- coding: utf-8 -*-
"""
Module 9: Text Cleaner - Advanced Sanitisation, Normalisation & OCR Repair.
Handles:
- HTML entities & tags removal
- URL and Email redacting
- Emoji stripping
- Contraction expansion
- CamelCase splitting (e.g. 'SevereChestPain' -> 'Severe Chest Pain')
- OCR noise correction (e.g. 'Amoxici11in' -> 'Amoxicillin', 'bord3rs' -> 'borders')
- PHI de-identification (names, MRNs, DOBs, phone numbers)
- Unicode NFKC normalisation & whitespace trimming
"""

import re
import html
import unicodedata
from typing import List

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}

# Common clinical OCR misreadings
OCR_CORRECTIONS = {
    r"\bAmoxici11in\b": "Amoxicillin",
    r"\bbord3rs\b": "borders",
    r"\b\|rregular\b": "irregular",
    r"\bpalp1tations\b": "palpitations",
    r"\btemp 1O1F\b": "temp 101F",
    r"\bpasi\b": "PASI",
}

URL_RE   = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+")
HTML_RE  = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\u2600-\u26FF\u2700-\u27BF]+",
    flags=re.UNICODE,
)
PHI_RE   = re.compile(r"MRN\d+|DOB:\s*\d{4}-\d{2}-\d{2}|Phone:\s*\+?[\d-]+|Email:\s*[\w.+-]+@[\w.-]+")
NAME_RE  = re.compile(r"(?:Patient|Name|Referred by Dr\.)\s*:?\s*[A-Z][a-z]+ [A-Z][a-z]+")

def _expand_contractions(text: str) -> str:
    for c, e in CONTRACTIONS.items():
        text = re.sub(c, e, text, flags=re.IGNORECASE)
    return text

def _split_camel_case(text: str) -> str:
    """Splits joined words like 'SevereChestPain' into 'Severe Chest Pain'."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    return text

def _repair_ocr(text: str) -> str:
    """Repairs common OCR scanner optical misreadings."""
    for pattern, replacement in OCR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

class TextCleaner:
    """Production-grade clinical text cleaner."""
    def __init__(self, remove_phi: bool = True, split_camel: bool = True, fix_ocr: bool = True):
        self.remove_phi = remove_phi
        self.split_camel = split_camel
        self.fix_ocr = fix_ocr

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = html.unescape(text)
        t = HTML_RE.sub(" ", t)
        t = URL_RE.sub(" ", t)
        t = EMAIL_RE.sub("[EMAIL]", t)
        t = EMOJI_RE.sub(" ", t)
        
        if self.split_camel:
            t = _split_camel_case(t)
            
        if self.fix_ocr:
            t = _repair_ocr(t)
            
        t = _expand_contractions(t)
        
        if self.remove_phi:
            t = PHI_RE.sub("[PHI]", t)
            t = NAME_RE.sub("[NAME]", t)
            
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]
