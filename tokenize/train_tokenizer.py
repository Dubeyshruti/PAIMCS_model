import os
import re
import argparse
import logging
import sentencepiece as spm
from datasets import load_dataset

# ----- CONFIGURE LOGGING -----
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    filename="train_tokenizer.log"
)
logger = logging.getLogger(__name__)

# Regex to split sentences on Hindi danda, punctuation (., !, ?)
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[\u0964.!?।])\s+')

def split_sentences(text: str):
    """
    Splits a text string into sentences using punctuation marks,
    including the Hindi danda (।), period, exclamation, or question marks.
    """
    if not text:
        return []
    text = text.replace('\u0964', '।')
    return [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def get_dataset_iterators():
    sources = []
    try:
        ds_fw = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True
        )
        sources.append(("FineWeb10BT", (ex.get("text", "") for ex in ds_fw)))
        logger.info("Loaded FineWeb-Edu sample-10BT (streaming).")
    except Exception as e:
        logger.error(f"Could not load FineWeb-Edu: {e}")
    try:
        ds_hi = load_dataset(
            "Hindi-data-hub/odaigen_hindi_pre_trained_sp",
            split="train",
            streaming=True
        )
        sources.append(("OdaigenHindi", (ex.get("content", "") for ex in ds_hi)))
        logger.info("Loaded Odaigen Hindi dataset (streaming).")
    except Exception as e:
        logger.error(f"Could not load Odaigen Hindi: {e}")
    return sources


def local_file_iterator(local_path: str):
    if not os.path.isfile(local_path):
        logger.warning(f"Local file not found: {local_path}")
        return
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def sentence_generator(local_file_path: str):
    for name, text_iter in get_dataset_iterators():
        for text in text_iter:
            for sent in split_sentences(text):
                yield sent
    yield from local_file_iterator(local_file_path)


def train_tokenizer(args):
    """
    Train a SentencePiece tokenizer using a disk-backed corpus file for efficiency.
    """

    # Step 1: Dump all sentences to a temp file
    temp_path = args.temp_corpus or "temp_corpus.txt"
    if not os.path.exists(temp_path):
        logger.info(f"Dumping sentences to disk at {temp_path}...")
        with open(temp_path, "w", encoding="utf-8") as out_f:
            for sent in sentence_generator(args.local_file):
                out_f.write(sent + "\n")
        logger.info("Sentence dumping completed.")
    else:
        logger.info(f"Temporary corpus file already exists at {temp_path}. Skipping corpus dump.")


    # Step 2: Train using file input to avoid Python iterator memory
    logger.info("Starting SentencePiece training via file input...")
    spm_args = (
        f"--input={temp_path}"
        f" --model_prefix={args.model_prefix}"
        f" --vocab_size={args.vocab_size}"
        f" --character_coverage={args.character_coverage}"
        f" --model_type={args.model_type}"
        f" --max_sentence_length={args.max_sentence_length}"
        f" --seed_sentencepiece_size={args.seed_sentencepiece_size}"
        " --train_extremely_large_corpus=1"
        " --shuffle_input_sentence=1"
        " --input_sentence_size=7896543"
        f" --unk_id={args.unk_id} --unk_piece={args.unk_piece}"
        f" --pad_id={args.pad_id} --pad_piece={args.pad_piece}"
        f" --bos_id={args.bos_id} --eos_id={args.eos_id}"
        f" --user_defined_symbols={args.user_defined_symbols}"
    )
    spm.SentencePieceTrainer.Train(spm_args)
    logger.info(f"Training completed. Models: {args.model_prefix}.model/.vocab")


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer over large multilingual sources."
    )
    parser.add_argument("--local-file", "-l", default="../../cleaned_ncert_sentences.txt",
                        help="Path to local text file (one sentence per line).")
    parser.add_argument("--model-prefix", "-p", default="paimcs_tokenizer",
                        help="Output prefix for .model and .vocab files.")
    parser.add_argument("--vocab-size", "-v", type=int, default=32109,
                        help="Size of the tokenizer vocabulary.")
    parser.add_argument("--character-coverage", "-c", type=float, default=0.9998,
                        help="Character coverage for SentencePiece.")
    parser.add_argument("--model-type", "-m", choices=["unigram","bpe","char","word"],
                        default="bpe",
                        help="Model algorithm to use (BPE is more memory-efficient).")
    parser.add_argument("--max-sentence-length", type=int, default=100000,
                        help="Max sentence length to avoid skipping lines.")
    parser.add_argument("--unk-id", type=int, default=0, help="ID for <unk> token.")
    parser.add_argument("--unk-piece", type=str, default="<unk>", help="Symbol for unknown token.")
    parser.add_argument("--pad-id", type=int, default=1, help="ID for <pad> token.")
    parser.add_argument("--pad-piece", type=str, default="<pad>", help="Symbol for padding token.")
    parser.add_argument("--bos-id", type=int, default=2, help="ID for begin-of-sentence token.")
    parser.add_argument("--eos-id", type=int, default=3, help="ID for end-of-sentence token.")
    parser.add_argument("--user-defined-symbols", type=str,
                        default="<ASST>,<USER>",
                        help="Comma-separated special symbols to add.")
    parser.add_argument("--temp-corpus", type=str,
                        default="temp_corpus.txt",
                        help="Path to write the temporary corpus dump.")
    parser.add_argument("--seed-sentencepiece-size", type=int, default=1000000,
                        help="Number of seed pieces to sample for initialization.")
    args = parser.parse_args()
    train_tokenizer(args)

if __name__ == "__main__":
    main()
