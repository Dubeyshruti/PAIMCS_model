from tensorflow import float16, cast, Tensor, range as Range, newaxis, pow as Pow, sin, cos, concat, shape
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    def __init__(self, hidden_dim: int, max_len: int = 5000, **kwargs) -> None:
        """
        Computes sinusoidal positional encodings.
        Args:
            d_model (int): Embedding dimension.
            max_len (int): Maximum sequence length.
        """
        super().__init__(dtype=float16, **kwargs)
        self.hidden_dim: int = hidden_dim
        pos_encoding: Tensor = self.get_positional_encoding(max_len, hidden_dim)
        self.pos_encoding: Tensor = cast(pos_encoding, float16)

    def get_positional_encoding(self, max_len: int, hidden_dim: int) -> Tensor:
        pos = Range(max_len, dtype=float16)[:, newaxis]  # (max_len, 1)
        i = Range(hidden_dim, dtype=float16)[newaxis, :]     # (1, d_model)
        angle_rates = 1 / Pow(10000, (2 * (i // 2)) / cast(hidden_dim, float16))
        angle_rads = pos * angle_rates  # (max_len, d_model)
        sines = sin(angle_rads[:, 0::2])
        cosines = cos(angle_rads[:, 1::2])
        pos_encoding = concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[newaxis, ...]  # (1, max_len, d_model)
        return pos_encoding

    def call(self, x: Tensor) -> Tensor:
        seq_len = shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]