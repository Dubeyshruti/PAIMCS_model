import re
import os
import unicodedata
import hashlib
from typing import List, Set
from datasets import load_dataset
import regex  # Supports Unicode script detection

def normalize_text(text: str) -> str:
    """
    Normalize unicode, collapse whitespace, strip boilerplate.
    """
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"\s+", ' ', text)
    return text.strip()

def clean_text(text: str) -> str:
    """
    Remove unwanted characters, references, and non-content artifacts.
    """
    # Remove page numbers, reprint notes, years
    text = re.sub(r"Unit\s+\d+.*?epress|Reprint.*?\d{4}", '', text)
    text = re.sub(r"\d{4}[-–]\d{2,4}", '', text)
    # Remove figure/table references and captions
    text = re.sub(r"Fig(?:ure)?\.\s*\d+[\.\d]*[A-Za-z\s,\(\)–-]*", '', text)
    # Remove bracketed citations like [1]
    text = re.sub(r"\[.*?\]", '', text)
    # Remove teaching prompts, keywords sections
    text = re.sub(r"(Questions\?|Activity \d+\.|Keywordssub|Let’s recall|Source \d+)", '', text)
    # Remove non-printable characters
    text = ''.join(ch for ch in text if ch.isprintable())
    return text.strip()

def split_sentences(texts: List[str]) -> List[str]:
    """
    Split on English [.?!] or Hindi [।] sentence-endings.
    """
    pat = re.compile(r'(?<=[\.!?।])\s+')
    sentences = []
    for doc in texts:
        for sent in pat.split(doc):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences

def filter_sentences(sentences: List[str], min_len: int = 2, max_len: int = 250) -> List[str]:
    """
    Filter English and Hindi sentences with separate rules as needed.
    """
    filtered = []
    for sent in sentences:
        words = sent.split()
        if len(words) < min_len or len(sent) > max_len:
            continue

        # Use regex to detect Devanagari
        is_hindi = bool(regex.search(r'\p{Script=Devanagari}', sent))

        # Use more lenient ratio for Hindi
        letter_ratio = sum(ch.isalpha() or ch.isspace() for ch in sent) / len(sent)
        if (is_hindi and letter_ratio < 0.5) or (not is_hindi and letter_ratio < 0.7):
            continue

        # Only apply ALL-CAPS heuristic to Latin script
        if not is_hindi:
            cap_words = [w for w in words if re.fullmatch(r'[A-Z]{2,}', w)]
            if len(cap_words) > 3:
                continue

        filtered.append(sent)
    return filtered

def deduplicate(sentences: List[str]) -> List[str]:
    """
    Remove exact duplicates via hashing.
    """
    seen: Set[str] = set()
    unique = []
    for sent in sentences:
        h = hashlib.md5(sent.encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(sent)
    return unique

def save_corpus(sentences: List[str], out_path: str):
    """
    Write one sentence per line to output file.
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent + '\n')

def is_devanagari(text):
    return bool(regex.search(r'\p{Script=Devanagari}', text))

if __name__ == '__main__':
    ds = load_dataset("ritik22912/ncert_upsc_text_6to12")
    cleaned_sentences_path = 'cleaned_ncert_sentences.txt'
    raw_texts =  [t for t in ds['train']['text']]
    normalized = [normalize_text(clean_text(t)) for t in raw_texts]
    sents = split_sentences(normalized)
    filtered = filter_sentences(sents)
    unique = deduplicate(filtered)
    save_corpus(unique, cleaned_sentences_path)

    print(f'Total raw docs: {len(raw_texts)}')
    print(f'Extracted sentences: {len(sents)}')
    print(f'After filtering: {len(filtered)}')
    print(f'Unique sentences: {len(unique)}')
