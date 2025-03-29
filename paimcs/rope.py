from typing import Tuple
import math
from tensorflow import Tensor, range as Range, float16, einsum, sin, cos, newaxis, shape, split, reshape, concat

def rotary_embedding(head_dim: int, seq_len: int, base: int = 12315) -> Tuple[Tensor, Tensor]:
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
    pos = Range(seq_len, dtype=float16)
    dim_range = Range(0, head_dim, 2, dtype=float16)
    inv_freq = 1.0 / (base ** (dim_range / head_dim))
    sinusoid_inp = einsum('i,j->ij', pos, inv_freq)  # (seq_len, head_dim//2)
    sine = sin(sinusoid_inp)[newaxis, ...]  # (1, seq_len, head_dim//2)
    cosine = cos(sinusoid_inp)[newaxis, ...]  # (1, seq_len, head_dim//2)
    return sine, cosine

def apply_rotary_pos_emb(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """
    Applies rotary positional embeddings.
    Args:
        x (tf.Tensor): Shape (batch, num_heads, seq_len, head_dim)
        sin, cos (tf.Tensor): Shapes (1, seq_len, head_dim//2)
    Returns:
        tf.Tensor: x with rotary encoding applied.
    """
    d_half = shape(x)[-1] // 2
    x1, x2 = split(x, num_or_size_splits=2, axis=-1)
    sin = reshape(sin, (1, 1, shape(sin)[1], d_half))
    cos = reshape(cos, (1, 1, shape(cos)[1], d_half))
    return concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)