from tensorflow import float16, cast, Tensor, range as Range, newaxis, pow as Pow, sin, cos, concat, shape
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, max_len: int = 5000, **kwargs) -> None:
        """
        Computes sinusoidal positional encodings.
        Args:
            d_model (int): Embedding dimension.
            max_len (int): Maximum sequence length.
        """
        super().__init__(dtype=float16, **kwargs)
        self.d_model: int = d_model
        pos_encoding: Tensor = self.get_positional_encoding(max_len, d_model)
        self.pos_encoding: Tensor = cast(pos_encoding, float16)

    def get_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        pos = Range(max_len, dtype=float16)[:, newaxis]  # (max_len, 1)
        i = Range(d_model, dtype=float16)[newaxis, :]     # (1, d_model)
        angle_rates = 1 / Pow(10000, (2 * (i // 2)) / cast(d_model, float16))
        angle_rads = pos * angle_rates  # (max_len, d_model)
        sines = sin(angle_rads[:, 0::2])
        cosines = cos(angle_rads[:, 1::2])
        pos_encoding = concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[newaxis, ...]  # (1, max_len, d_model)
        return pos_encoding

    def call(self, x: Tensor) -> Tensor:
        seq_len = shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]