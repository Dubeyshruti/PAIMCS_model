from tensorflow import Tensor, cast, shape, float16, matmul, sqrt, split, concat, reshape, transpose
from tensorflow.nn import softmax
from tensorflow.keras import layers
from tensorflow.random import normal
from rope import apply_rotary_pos_emb

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Computes scaled dot–product attention with rotary embeddings. No masking is applied."""
    q = apply_rotary_pos_emb(q, sin, cos)
    k = apply_rotary_pos_emb(k, sin, cos)
    dk = cast(shape(k)[-1], float16)
    scaled_logits = matmul(q, k, transpose_b=True) / sqrt(dk)
    attn_weights = softmax(scaled_logits, axis=-1)
    return matmul(attn_weights, v)

class GroupedPointwiseConv(layers.Layer):
    def __init__(self, filters: int, groups: int = 3, **kwargs):
        super().__init__(dtype=float16, **kwargs)
        if filters % groups != 0:
            raise ValueError("filters must be divisible by groups")
        self.groups = groups
        self.group_convs = [
            layers.Conv1D(filters // groups, kernel_size=1, padding='same', use_bias=False, dtype='float16')
            for _ in range(groups)
        ]

    def call(self, inputs: Tensor) -> Tensor:
        return concat(
            [conv(x) for conv, x in zip(self.group_convs, split(inputs, self.groups, axis=-1))],
            axis=-1
        )

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model: int = 360, num_heads: int = 3, groups: int = 3, **kwargs):
        super().__init__(dtype=float16, **kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads  # per-head dimension

        self.wq = GroupedPointwiseConv(d_model, groups)
        self.wk = GroupedPointwiseConv(d_model, groups)
        self.wv = GroupedPointwiseConv(d_model, groups)
        self.dense = GroupedPointwiseConv(d_model, groups)

    def split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        # Reshape x from (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, depth), then transpose to (batch_size, num_heads, seq_len, depth)
        return transpose(reshape(x, (batch_size, -1, self.num_heads, self.depth)), perm=[0, 2, 1, 3])

    def call(self, v: Tensor, k: Tensor, q: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        batch_size = shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        attn_out = transpose(scaled_dot_product_attention(q, k, v, sin, cos), perm=[0, 2, 1, 3])
        return self.dense(reshape(attn_out, (batch_size, -1, self.d_model)))

def main() -> None:
    # Parameters
    batch_size = 2; seq_len = 10; d_model = 360; num_heads = 3; groups = 3
    depth = d_model // num_heads    # For MultiHeadAttention, depth = 360/3 = 120
    d_half = depth // 2             # Each rotary embedding half: 120/2 = 60

    # --- Test Scaled Dot–Product Attention  ---
    # Here, we assume inputs are already split into heads:- Shape: (batch_size, num_heads, seq_len, depth)
    q_att = normal((batch_size, num_heads, seq_len, depth), dtype=float16)
    k_att = normal((batch_size, num_heads, seq_len, depth), dtype=float16)
    v_att = normal((batch_size, num_heads, seq_len, depth), dtype=float16)
    # Rotary embeddings shape: (1, 1, seq_len, d_half)
    sin_att = normal((1, 1, seq_len, d_half), dtype=float16)
    cos_att = normal((1, 1, seq_len, d_half), dtype=float16)
    attn_output = scaled_dot_product_attention(q_att, k_att, v_att, sin_att, cos_att)
    print("Scaled Dot Product Attention output shape:", attn_output.shape)

    # --- Test Grouped Pointwise Convolution ---
    grouped_conv = GroupedPointwiseConv(d_model, groups)
    conv_input = normal((batch_size, seq_len, d_model), dtype=float16)
    conv_output = grouped_conv(conv_input)
    print("GroupedPointwiseConv output shape:", conv_output.shape)

    # --- Test MultiHeadAttention ---
    mha = MultiHeadAttention(d_model, num_heads, groups)
    # Full inputs have shape: (batch_size, seq_len, d_model)
    q_input = normal((batch_size, seq_len, d_model), dtype=float16)
    k_input = normal((batch_size, seq_len, d_model), dtype=float16)
    v_input = normal((batch_size, seq_len, d_model), dtype=float16)
    # Rotary embeddings for MHA: shape (1, 1, seq_len, d_half)
    sin_mha = normal((1, 1, seq_len, d_half), dtype=float16)
    cos_mha = normal((1, 1, seq_len, d_half), dtype=float16)
    mha_output = mha(v_input, k_input, q_input, sin_mha, cos_mha)
    print("MultiHeadAttention output shape:", mha_output.shape)

if __name__ == "__main__":
    main()
