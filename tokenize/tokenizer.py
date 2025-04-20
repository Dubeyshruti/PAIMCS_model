import sentencepiece as spm
import os

def train_sentencepiece(corpus_file: str,
                        model_prefix: str = "paimcs_tokenizer",
                        vocab_size: int = 2673,
                        model_type: str = "unigram",
                        character_coverage: float = 1.0,
                        user_defined_symbols: list[str] = None):
    """
    Train a SentencePiece tokenizer.

    Args:
      corpus_file: Path to a newline‐delimited training corpus.
      model_prefix: Prefix for the output model files (will create .model & .vocab).
      vocab_size: Number of tokens in the vocabulary.
      model_type: One of 'unigram', 'bpe', 'word', or 'char'.
      character_coverage: Amount of characters covered by the model (for languages like Japanese, use <1.0).
      user_defined_symbols: List of extra symbols to include (e.g. ['<s>', '</s>']).
    """
    cmd = (f"--input={corpus_file} "
           f"--model_prefix={model_prefix} "
           f"--vocab_size={vocab_size} "
           f"--model_type={model_type} "
           f"--character_coverage={character_coverage}"
           " --pad_id=3 --bos_id=1 --eos_id=2 --unk_id=0"
           " --max_sentence_length=10000")
    if user_defined_symbols:
        symbols = ",".join(user_defined_symbols)
        cmd += f" --user_defined_symbols={symbols}"
    spm.SentencePieceTrainer.Train(cmd)
    print(f"Trained SentencePiece model: {model_prefix}.model / {model_prefix}.vocab")

def load_sentencepiece(model_file: str):
    """
    Load a trained SentencePiece model.

    Returns:
      A SentencePieceProcessor instance with encode/decode methods.
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    return sp

if __name__ == "__main__":
    # 1. Train (once):
    train_sentencepiece(
        corpus_file="../../tokenizer_train_input.txt",
        model_prefix="paimcs_tokenizer",
        vocab_size=2673,
        model_type="unigram",
        character_coverage=1.0
    )

    # 2. Load and use:
    sp = load_sentencepiece("paimcs_tokenizer.model")

    # Encode a sentence to token IDs
    text = "The quick brown fox jumps over the lazy dog."
    ids = sp.EncodeAsIds(text)
    pieces = sp.EncodeAsPieces(text)
    print("IDs:   ", ids)
    print("Pieces:", pieces)

    # Decode back
    decoded = sp.DecodeIds(ids)
    print("Decoded:", decoded)
