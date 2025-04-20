from datasets import load_dataset
import re
from os import makedirs
ds = load_dataset("ritik22912/ncert_upsc_text_6to12")
HI_save_dir = '../../HinSeedText'
EN_save_dir = '../../EnSeedText'
makedirs(HI_save_dir, exist_ok=True)
makedirs(EN_save_dir, exist_ok=True)

def clean_text(raw_text: str) -> str:
    """
    Step 1: Clean extracted OCR/text-extracted content:
      - Strip timestamps, standalone page numbers, URLs, captions, and noise lines.
      - De-hyphenate split words.
      - Merge broken lines into coherent blocks.
    """
    # Remove form‑feeds and zero‑width joiners
    text = raw_text.replace("\x0c", " ").replace("\u200d", "")
    # Remove timestamps
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}", "", text)
    # Remove standalone page numbers
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove lines with only non-word chars or underscores
    text = re.sub(r"(?m)^[\W_]{5,}$", "", text)
    # Remove caption lines
    text = re.sub(r"(?m)^\s*चित्र.*$", "", text)
    # De-hyphenate broken words at line ends
    text = re.sub(r"(\w+)-\n(\w+)", lambda m: m.group(1) + m.group(2), text)

    # Merge lines based on punctuation and casing
    lines = text.splitlines()
    merged_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            merged_lines.append('')
            continue
        # Skip residual numeric/OCR artifacts
        if re.match(r"^[\d\[\]#]+$", s):
            continue
        if re.match(r'^\s*(\d+|Unit\s+\d+|Reprint|\d{2}-\d{2}-\d{4})', s):
            continue
        s = re.sub(r'[_\-\|]{2,}', '', s)             # drop long punctuation runs
        s = re.sub(r'\s{2,}', ' ', s).strip()          # collapse spaces
        if merged_lines and not merged_lines[-1].endswith(('.', '?', '!', ':')) and s and s[0].islower():
            merged_lines[-1] += ' ' + s
        else:
            merged_lines.append(s)
    # Collapse multiple blank lines
    cleaned = []
    for ln in merged_lines:
        if ln == '' and cleaned and cleaned[-1] == '':
            continue
        cleaned.append(ln)
    return '\n'.join(cleaned)


def format_readable(cleaned_text: str) -> str:
    """
    Step 2: Apply readability formatting:
      - Split into paragraphs at blank lines.
      - Within each paragraph, join any remaining line breaks.
      - Detect headings (<=5 words) and wrap with **bold**.
      - Detect questions (ending with '?') and prefix with a bullet.
      - Leave other paragraphs as plain text.
    """
    paras = re.split(r"(?:\r?\n){2,}", cleaned_text)
    formatted = []
    for para in paras:
        p = ' '.join(line.strip() for line in para.splitlines() if line.strip())
        if not p:
            continue
        words = p.split()
        if len(words) <= 5:
            # Treat as heading
            formatted.append(f"**{p}**")
        elif p.endswith('?'):
            formatted.append(f"- {p}")
        else:
            formatted.append(p)
    return '\n\n'.join(formatted)

for pdfName, rawText in zip(ds['train']['pdfName'], ds['train']['text']):
    if '_h_' in pdfName:
        with open(f"{HI_save_dir}/{pdfName}.txt", 'w', encoding='utf-8') as f:
            f.write(format_readable(clean_text(rawText)))
    else:
        with open(f"{EN_save_dir}/{pdfName}.txt", 'w', encoding='utf-8') as f:
            f.write(format_readable(clean_text(rawText)))
