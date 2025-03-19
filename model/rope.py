from tensorflow import Tensor, cast, range as Range, int32, float16, einsum, expand_dims, sin as Sin, cos as Cos, shape, split, reshape, concat
from tensorflow.random import uniform

def rotary_embedding(head_dim: int = 120, seq_len: int = 256, base: int = 12315) -> tuple[Tensor, Tensor]:
    """ Returns: Sinusoidal embeddings (sin, cos) of shape (1, seq_len, head_dim//2)"""
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be an even integer.")

    inv_freq = 1.0 / (base ** (cast(Range(0, head_dim, 2, dtype=int32), float16) / cast(head_dim, float16)))
    sinusoid_inp = einsum('i,j->ij', cast(Range(seq_len), float16), inv_freq)
    return expand_dims(Sin(sinusoid_inp), axis=0), expand_dims(Cos(sinusoid_inp), axis=0) # Shape : (1, seq_len, head_dim//2)

def apply_rotary_pos_emb(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """  Args:
        x (Tensor): shape (batch, num_heads, seq_len, head_dim)
        sin (Tensor): Sine values of shape (1, seq_len, head_dim//2)
        cos (Tensor): Cosine values of shape (1, seq_len, head_dim//2)
    """
    d_half = shape(x)[-1] // 2
    x1, x2 = split(x, num_or_size_splits=2, axis=-1)
    sin, cos = reshape(sin, (1, 1, -1, d_half)), reshape(cos, (1, 1, -1, d_half)) # Reshape sin and cos to (1, 1, seq_len, d_half)
    return concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

def get_int_input(prompt: str, default: int) -> int:
    """ Gets an integer input from the user with a default fallback."""
    try:
        return int(input(f"{prompt} (default: {default}): ").strip())
    except ValueError:
        print(f"Invalid input. Using default value: {default}")
        return default

def main() -> None:
    head_dim = get_int_input('Enter Dimension for Rotary Embedding calculation (must be even)=', 120)
    seq_len = get_int_input('Enter a Sequence length value=', 256)
    base = get_int_input('Enter base value for calculating frequencies=', 12315)

    x = uniform(shape=(1, seq_len, head_dim), dtype=float16)
    print(f"Input: {x}")

    sin, cos = rotary_embedding(head_dim, seq_len, base)
    print("SIN and COS values:", sin, cos, sep='\n', end='\n\n')

    print("Output after applying rotary position embeddings", apply_rotary_pos_emb(x, sin, cos), sep = '\n')

if __name__ == "__main__":
    main()
