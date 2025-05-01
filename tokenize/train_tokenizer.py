import os
import re
import argparse
import logging
import sentencepiece as spm
from datasets import load_dataset
from itertools import chain

# ----- CONFIGURE LOGGING -----
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    filename='train_tokenizer.log'
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
    # Normalize whitespace
    text = text.replace('\u0964', '।')  # ensure consistent danda
    return [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def get_dataset_iterators():
    """
    Return iterators over raw text from streaming HF datasets.
    """
    sources = []
    # FineWeb-Edu (10BT sample)
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

    # Odaigen Hindi
    try:
        ds_hi = load_dataset(
            "Hindi-data-hub/odaigen_hindi_pre_trained_sp",
            split="train",
            streaming=True
        )
        sources.append(("OdaigenHindi", (ex.get("text", "") for ex in ds_hi)))
        logger.info("Loaded Odaigen Hindi dataset (streaming).")
    except Exception as e:
        logger.error(f"Could not load Odaigen Hindi: {e}")

    return sources


def local_file_iterator(local_path: str):
    """
    Yields each line from a local UTF-8 file as one sentence.
    """
    if not os.path.isfile(local_path):
        logger.warning(f"Local file not found: {local_path}")
        return
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def sentence_generator(local_file_path: str):
    """
    Yield all sentences from datasets and local file, splitting as needed.
    """
    # Streaming datasets
    for name, text_iter in get_dataset_iterators():
        for text in text_iter:
            for sent in split_sentences(text):
                yield sent

    # Local file (one sentence per line)
    yield from local_file_iterator(local_file_path)


def train_tokenizer(args):
    """
    Train a SentencePiece tokenizer using the sentence_generator over the full corpus.
    """
    logger.info("Starting SentencePiece training on full corpus,,,")
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_generator(args.local_file),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        unk_id=args.unk_id,
        unk_piece=args.unk_piece,
        pad_id=args.pad_id,
        pad_piece=args.pad_piece,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        user_defined_symbols=args.user_defined_symbols.split(","),
        shuffle_input_sentence=False  # use full data in order
    )
    logger.info(f"Training completed. Models: {args.model_prefix}.model/.vocab")


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer over large multilingual sources."
    )
    parser.add_argument(
        "--local-file", "-l",
        default="../../cleaned_ncert_sentences.txt",
        help="Path to local text file (one sentence per line)."
    )
    parser.add_argument(
        "--model-prefix", "-p",
        default="paimcs_tokenizer",
        help="Output prefix for .model and .vocab files."
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int, default=32109,
        help="Size of the tokenizer vocabulary."
    )
    parser.add_argument(
        "--character-coverage", "-c",
        type=float, default=0.999987654321,
        help="Character coverage for SentencePiece (good for Hindi+English)."
    )
    parser.add_argument(
        "--model-type", "-m",
        choices=["unigram","bpe","char","word"],
        default="unigram",
        help="Model algorithm to use."
    )
    # Special token arguments
    parser.add_argument("--unk-id", type=int, default=0, help="ID for <unk> token.")
    parser.add_argument("--unk-piece", type=str, default="<unk>", help="Symbol for unknown token.")
    parser.add_argument("--pad-id", type=int, default=1, help="ID for <pad> token.")
    parser.add_argument("--pad-piece", type=str, default="<pad>", help="Symbol for padding token.")
    parser.add_argument("--bos-id", type=int, default=2, help="ID for begin-of-sentence token.")
    parser.add_argument("--eos-id", type=int, default=3, help="ID for end-of-sentence token.")
    parser.add_argument(
        "--user-defined-symbols", type=str,
        default="<ASST>,<USER>",
        help="Comma-separated special symbols to add."
    )

    args = parser.parse_args()
    train_tokenizer(args)

if __name__ == "__main__":
    main()
