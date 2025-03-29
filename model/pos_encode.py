# =========================================
# Sinusoidal Positional Encoding (external, absolute positions)
# =========================================
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, max_len: int = 5000, **kwargs) -> None:
        """
        Computes sinusoidal positional encodings.
        Args:
            d_model (int): Embedding dimension.
            max_len (int): Maximum sequence length.
        """
        super().__init__(**kwargs)
        self.d_model: int = d_model
        pos_encoding = self.get_positional_encoding(max_len, d_model)
        self.pos_encoding: tf.Tensor = tf.cast(pos_encoding, tf.float32)

    def get_positional_encoding(self, max_len: int, d_model: int) -> tf.Tensor:
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]  # (max_len, 1)
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]     # (1, d_model)
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates  # (max_len, d_model)
        # apply sin to even indices; cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, max_len, d_model)
        return pos_encoding

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

# =========================================
# Rotary Positional Encoding Functions (for relative positions)
# =========================================
def rotary_embedding(head_dim: int, seq_len: int, base: int = 12315) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes rotary positional embeddings (sin, cos).
    Args:
        head_dim (int): Must be even.
        seq_len (int): Sequence length.
        base (int): Frequency base.
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: sin and cos embeddings of shape (1, seq_len, head_dim//2)
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even.")
    pos = tf.range(seq_len, dtype=tf.float32)  # (seq_len,)
    dim_range = tf.range(0, head_dim, 2, dtype=tf.float32)  # (head_dim//2,)
    inv_freq = 1.0 / (base ** (dim_range / head_dim))
    sinusoid_inp = tf.einsum('i,j->ij', pos, inv_freq)  # (seq_len, head_dim//2)
    sin = tf.sin(sinusoid_inp)[tf.newaxis, ...]  # (1, seq_len, head_dim//2)
    cos = tf.cos(sinusoid_inp)[tf.newaxis, ...]  # (1, seq_len, head_dim//2)
    return sin, cos

def apply_rotary_pos_emb(x: tf.Tensor, sin: tf.Tensor, cos: tf.Tensor) -> tf.Tensor:
    """
    Applies rotary positional embeddings.
    Args:
        x (tf.Tensor): Shape (batch, num_heads, seq_len, head_dim)
        sin, cos (tf.Tensor): Shapes (1, seq_len, head_dim//2)
    Returns:
        tf.Tensor: x with rotary encoding applied.
    """
    d_half = tf.shape(x)[-1] // 2
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    sin = tf.reshape(sin, (1, 1, tf.shape(sin)[1], d_half))
    cos = tf.reshape(cos, (1, 1, tf.shape(cos)[1], d_half))
    return tf.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)