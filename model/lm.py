from tensorflow.keras import backend, Model, layers
from numpy import argmax
from tensorflow import float16, convert_to_tensor, int32, shape, Tensor
from transformers import TransformerBlock
from rope import rotary_embedding
from numpy.random import randint

# Set global precision
backend.set_floatx('float16')

class PAIMCS_lm(Model):
    def __init__(self, num_layers: int = 27, d_model: int = 360, num_heads: int = 3, dff: int = 1440, vocab_size: int = 31542, rate: float = 0.099, groups: int = 3, tokenizer=None, **kwargs):
        """ Large Language Model for multi-token prediction.
        Args:
            num_layers (int): Number of transformer layers.
            d_model (int): Model dimensionality.
            num_heads (int): Number of attention heads.
            dff (int): Dimensionality of the feed-forward network.
            vocab_size (int): Size of the vocabulary.
            rate (float): Dropout rate.
            groups (int): Number of groups for grouped pointwise convolutions.
            tokenizer: A SentencePieceProcessor instance providing eos_id() and eor_id() methods.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(dtype=float16, **kwargs)
        self.d_model = d_model; self.num_heads = num_heads; self.vocab_size = vocab_size; self.tokenizer = tokenizer
        self.embedding = layers.Embedding(vocab_size, d_model, dtype='float16')
        self.enc_layers = [TransformerBlock(d_model, num_heads, dff, rate, groups) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate, dtype = float16)
        self.final_layer = layers.Dense(vocab_size, dtype='float16')

    def call(self, x: Tensor) -> Tensor:
        """  x (Tensor): Tensor of shape (batch_size, seq_len) containing token IDs.
        Returns= Tensor: Logits over the vocabulary with shape (batch_size, seq_len, vocab_size). """
        seq_len = shape(x)[1]
        head_dim = self.d_model // self.num_heads
        # Generate rotary embeddings for the current sequence length.
        sin, cos = rotary_embedding(head_dim, seq_len)
        x = self.dropout(self.embedding(x))
        for layer in self.enc_layers:
            x = layer(x, sin, cos)
        return self.final_layer(x)

    def generate(self, prompt: str, max_new_tokens: int=64, max_total: int=1024) -> str:
        """ Autoregressively generate tokens. The model predicts up to max_new_tokens in one forward pass. The generated block is
        scanned for the first occurrence of <eos> (end-of-sequence); if found, that block is truncated. The new tokens are appended
        to the prompt. Generation stops when an <eor> (end-of-response) token is produced after an <eos>, or when max_total tokens
        are reached.
        Args:
            prompt (str): The prompt string.
            max_new_tokens (int): Maximum tokens to predict in one pass.
            max_total (int): Maximum total sequence length.
        Returns= str: The decoded generated text."""
        if self.tokenizer is None:
            raise ValueError("A tokenizer with eos_id() and eor_id() is required for generation.")
        eos_id = self.tokenizer.eos_id()
        eor_id = self.tokenizer.eor_id()
        token_ids = self.tokenizer.encode(prompt, out_type=int)
        prompt_length = len(token_ids)
        while len(token_ids) < max_total:
            cur_len = len(token_ids)
            # Pad the input with zeros for the new tokens.
            padded = token_ids + [0] * max_new_tokens
            inp_tensor = convert_to_tensor([padded], dtype=int32)
            logits = self.call(inp_tensor).numpy()[0]
            # Extract predictions for the new block.
            new_block = [int(argmax(logits[i])) for i in range(cur_len, cur_len + max_new_tokens)]
            if eos_id in new_block:
                eos_index = new_block.index(eos_id)
                token_ids.extend(new_block[:eos_index + 1])
                # Check if the token following <eos> is <eor> and stop if so.
                if eos_index + 1 < len(new_block) and new_block[eos_index + 1] == eor_id:
                    token_ids.append(eor_id)
                    break
            else:
                token_ids.extend(new_block)
            if len(token_ids) >= max_total:
                break
        return self.tokenizer.decode(token_ids[prompt_length:])

    def __call__(self, x, max_new_tokens: int = 64, max_total: int = 1024):
        """  Overrides the default __call__ to support string inputs. If x is a string, it is treated as a prompt for generation.
        Otherwise, x is assumed to be a tensor and a normal forward pass is performed.
        Args= x (str or Tensor): Input prompt or tensor.
        Returns= str or Tensor: Generated text if x is a string; otherwise, model logits."""
        if isinstance(x, str):
            return self.generate(x, max_new_tokens, max_total)
        else:
            return self.call(x)

def main() -> None:
    # Define a dummy tokenizer for testing purposes.
    class DummyTokenizer:
        def __init__(self):
            self._vocab = {"<eos>": 1, "<eor>": 2}; self._id_to_word = {1: "<eos>", 2: "<eor>"} ; self.next_id = 3
        def eos_id(self):
            return 1
        def eor_id(self):
            return 2

        def encode(self, prompt, out_type=int):
            # A simple encoding: split on whitespace and assign an id for each word.
            tokens = []
            for word in prompt.split():
                if word not in self._vocab:
                    self._vocab[word] = self.next_id
                    self._id_to_word[self.next_id] = word
                    self.next_id += 1
                tokens.append(self._vocab[word])
            tokens.append(self.eos_id())  # Append <eos> token.
            return tokens

        def decode(self, tokens):
            # A simple decoding: convert token ids back to words.
            words = [self._id_to_word.get(token, "<unk>") for token in tokens]
            return " ".join(words)

    # Instantiate the dummy tokenizer.
    tokenizer = DummyTokenizer()

    # Create a small LLM model instance for testing.
    model = PAIMCS_lm(num_layers=2, d_model=360, num_heads=3, dff=1440, vocab_size=22, tokenizer=tokenizer)

    # Test forward pass with tokenized input.
    batch_size = 2
    seq_len = 10
    # Create a random tensor of token IDs (range: 0 to vocab_size-1).
    input_ids = randint(0, 22, size=(batch_size, seq_len))
    input_tensor = convert_to_tensor(input_ids, dtype=int32)
    logits = model(input_tensor)
    print("Forward pass output shape:", logits.shape)

    # Test generation with a string prompt.
    prompt = "hello world this is a test"
    generated_text = model(prompt, max_new_tokens = 2, max_total = 4)
    print("Generated output:", generated_text)

if __name__ == "__main__":
    main()
