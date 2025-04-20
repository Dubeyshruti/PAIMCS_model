import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# --- CLI args ---
parser = argparse.ArgumentParser(
    description="Generate synthetic Civil Services training corpus"
)
parser.add_argument(
    "language",
    choices=["en", "hi"],
    help="Language to process: 'en' (English) or 'hi' (Hindi)"
)
args = parser.parse_args()
LANG = args.language

# --- Paths & model choices ---
OUTPUT_DIR   = "../../PretrainCorpus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if LANG == "en":
    SEED_DIR      = "../../EnSeedText"
    MODEL_NAME    = "mistralai/Mistral-7B-Instruct-v0.1"
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_en.json")
elif LANG == "hi":
    SEED_DIR      = "../../HinSeedText"
    MODEL_NAME    = "sarvamai/sarvam-1"
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_hi.json")

# --- Chunking & batch size ---
MAX_WORDS  = 221
OVERLAP    = 56
BATCH_SIZE = 8

# --- Generation hyperparameters ---
GEN_KWARGS = {
    "max_new_tokens": 234,
    "do_sample": True,
    "top_p": 0.899,
    "temperature": 0.687,
    "repetition_penalty": 1.098,
}

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("⚠️  Running on CPU; very slow. GPU highly recommended.")

# --- Load tokenizer & (quantized) model ---
print(f"Loading {LANG.upper()} model: {MODEL_NAME} ...")

if LANG == "en":
    # 4-bit quantization config for Mistral
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# --- Helper: chunk text into overlapping windows ---
def chunk_text(text, max_words=MAX_WORDS, overlap=OVERLAP):
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

# --- Batched generation ---
@torch.no_grad()
def generate_batch(model, tokenizer, chunks: list[str]):
    """Batch-generate for multiple chunks in one forward pass."""
    # choose prompt template
    if LANG == "en":
        tmpl = "<s>[INST] Summarize and elaborate this for a UPSC aspirant:\n{chunk}\n[/INST]"
    else:
        tmpl = "<s>[INST] निम्नलिखित पाठ का संक्षेप करें और एक विस्तृत, पराफ्रेज़्ड प्रशिक्षण अनुच्छेद बनाएं:\n{chunk}\n[/INST]"
    prompts = [tmpl.format(chunk=ch.strip()) for ch in chunks]

    # tokenize & move to device
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # generate
    out_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        **GEN_KWARGS
    )

    # decode
    return [
        tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
        for ids, input_len in zip(
            out_ids,
            (inputs["input_ids"].ne(tokenizer.pad_token_id)
                          .sum(dim=1).tolist())
        )
    ]

# --- File processing with checkpointing & batching ---
def process_file(file_path, model, tokenizer, progress):
    fname    = os.path.basename(file_path)
    out_path = os.path.join(OUTPUT_DIR, fname)

    # read & chunk
    text   = open(file_path, "r", encoding="utf-8").read().strip()
    chunks = chunk_text(text)
    total  = len(chunks)
    start  = progress.get(fname, 0)
    if start >= total:
        print(f"✔️  {fname} already done, skipping.")
        return

    with open(out_path, "a", encoding="utf-8") as out_f:
        # batch loop
        for batch_start in range(start, total, BATCH_SIZE):
            batch_idxs   = list(range(batch_start, min(batch_start+BATCH_SIZE, total)))
            batch_chunks = [chunks[i] for i in batch_idxs]
            outputs      = generate_batch(model, tokenizer, batch_chunks)

            # write & checkpoint per chunk
            for idx, text_out in zip(batch_idxs, outputs):
                out_f.write(f"# Chunk {idx}\n{text_out}\n\n")
                out_f.flush()
                progress[fname] = idx + 1
                with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
                    json.dump(progress, pf, indent=2, ensure_ascii=False)
                print(f"✅ {fname} chunk {idx+1}/{total} done.")

# --- Main execution ---
def main():
    # load or initialize progress file
    if os.path.exists(PROGRESS_FILE):
        progress = json.load(open(PROGRESS_FILE, "r", encoding="utf-8"))
    else:
        progress = {}

    seed_files = [
        os.path.join(SEED_DIR, f)
        for f in os.listdir(SEED_DIR)
        if f.endswith(".txt")
    ]

    for fp in tqdm(seed_files, desc=f"▸ {LANG.upper()} seeds"):
        process_file(fp, model, tokenizer, progress)

    print("🎉 All done!")

if __name__ == "__main__":
    main()
