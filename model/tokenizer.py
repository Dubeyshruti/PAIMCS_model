from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from argparse import ArgumentParser

def train_sentencepiece_model(input_file: str, model_prefix: str="tokenizer", vocab_size: int = 31542, character_coverage: float = 1.0) -> None:
    """ Train a SentencePiece model on the given input file. The model will include user-defined symbols <eos> and <eor>.  Available input ways->
    - spm_train --input=corpus1.txt,corpus2.txt,corpus3.txt --model_prefix=spm_model --vocab_size=32000 --model_type=bpe
    - spm_train --input=corpus_list.txt --model_prefix=spm_model --vocab_size=32000 --model_type=bpe
    - cat corpus1.txt corpus2.txt corpus3.txt | spm_train --input=/dev/stdin --model_prefix=spm_model --vocab_size=32000 --model_type=bpe
    """
    SentencePieceTrainer.Train(f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage={character_coverage} --model_type=bpe --user_defined_symbols=<eos>,<eor>")

class SentencePieceTokenizer:
    """ A wrapper for the SentencePieceProcessor that exposes a simple tokenizer interface. This tokenizer support special tokens <eos> and <eor> for end-of-sequence and end-of-response."""
    def __init__(self, model_file: str):
        self.sp = SentencePieceProcessor()
        if not self.sp.Load(model_file):
            raise ValueError(f"Could not load SentencePiece model from {model_file}")

    def eos_id(self) -> int:
        eid = self.sp.PieceToId("<eos>")
        if eid < 0:
            raise ValueError("The SentencePiece model does not define an <eos> token.")
        return eid

    def eor_id(self) -> int:
        eid = self.sp.PieceToId("<eor>")
        if eid < 0:
            raise ValueError("The SentencePiece model does not define an <eor> token.")
        return eid

    def encode(self, text: str, out_type=int) -> list:
        return [out_type(x) for x in self.sp.EncodeAsIds(text)]

    def decode(self, token_ids: list) -> str:
        return self.sp.DecodeIds(token_ids)

def main() -> None:
    parser = ArgumentParser(description="Train or use a SentencePiece model for tokenization.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train a SentencePiece model.")
    train_parser.add_argument("--input_file", type=str, required=True, help="Path to the training corpus.")
    train_parser.add_argument("--model_prefix", type=str, default='tokenizer', help="Prefix for the output model files.")
    train_parser.add_argument("--vocab_size", type=int, default=31542, help="Vocabulary size (default: 31542).")
    train_parser.add_argument("--character_coverage", type=float, default=1.0, help="Character coverage (default: 1.0).")
    # Subparser for tokenizing a string
    tokenize_text_parser = subparsers.add_parser("tokenize", help="Tokenize a single string using a trained SentencePiece model.")
    tokenize_text_parser.add_argument("--model_file", type=str, required=True, help="Path to the trained SentencePiece model (.model).")
    tokenize_text_parser.add_argument("--text", type=str, required=True, help="Input text string to tokenize.")
    args = parser.parse_args()
    if args.command == "train":
        train_sentencepiece_model(args.input_file, args.model_prefix, args.vocab_size, args.character_coverage)
    else:
        tokenizer = SentencePieceTokenizer(args.model_file)
        print("Tokenized Output=", tokenizer.encode(args.text), sep='\t')
if __name__=="__main__":
    main()
