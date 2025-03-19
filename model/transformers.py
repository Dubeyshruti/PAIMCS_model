from tensorflow import float16, TensorShape, Tensor, sqrt, reduce_mean, square
from tensorflow.nn import softmax
from tensorflow.keras import layers, Sequential
from tensorflow.random import normal
from attention import MultiHeadAttention, GroupedPointwiseConv
from rope import rotary_embedding, apply_rotary_pos_emb
from tensorflow.random import normal

class RMSNorm(layers.Layer):
    def __init__(self, epsilon: float = 8.5e-6, **kwargs) -> None:
        """ epsilon: Small constant to prevent division by zero. """
        super().__init__(dtype=float16, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape: tuple) -> None:
        input_shape = TensorShape(input_shape)
        if input_shape.rank is None or input_shape[-1] is None:
            raise ValueError("The last dimension of the input shape must be defined.")
        # Create a trainable scale parameter gamma with the shape of the last dimension.
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer='ones', trainable=True, dtype='float16')
        super().build(input_shape)

    def call(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input x must be a tf.Tensor.")
        # Compute RMS over the last dimension and add epsilon for numerical stability.
        rms = sqrt(reduce_mean(square(x), axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * self.gamma

class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int = 360, num_heads: int = 3, dff: int = 1440, rate: float = 99e-3, groups: int = 3, **kwargs) -> None:
        super().__init__(dtype=float16, **kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads, groups)
        self.ffn = Sequential([GroupedPointwiseConv(dff, groups), layers.Activation('silu', dtype='float16'),
                               GroupedPointwiseConv(d_model, groups)])
        self.norm1 = RMSNorm(epsilon=8.5e-6); self.norm2 = RMSNorm(epsilon=8.5e-6)
        self.dropout1 = layers.Dropout(rate, dtype = float16); self.dropout2 = layers.Dropout(rate, dtype = float16)

    def call(self, x: Tensor, sin: Tensor, cos: Tensor, training: bool = False) -> Tensor:
        """x: Input tensor of shape (batch_size, seq_len, d_model). training: Boolean flag for training mode (affects dropout behavior). """

        out1 = self.norm1(x + self.dropout1(self.mha(x, x, x, sin, cos), training=training))
        ffn_output = self.dropout2(self.ffn(out1, training=training), training=training)
        return self.norm2(out1 + ffn_output)

def main() -> None:
    print("Testing RMSNorm...")
    batch_size, seq_len, d_model = 2, 5, 360
    x = normal((batch_size, seq_len, d_model), dtype=float16)
    norm_layer = RMSNorm(epsilon=8.5e-6)
    y_norm = norm_layer(x)
    print("RMSNorm output shape:", y_norm.shape)

    print("Testing TransformerBlock...")
    transformer_block = TransformerBlock(d_model=d_model, num_heads=3, dff=1440, rate=0.099, groups=3)
    # For MultiHeadAttention inside TransformerBlock, assume rotary embeddings are of shape:
    # (1, 1, num_heads, d_model//num_heads). Here, d_model//num_heads = 360//3 = 120.
    num_heads = 3
    depth = d_model // num_heads  # 120
    # Create sin and cos tensors with shape (1, 1, seq_len, depth)
    sin, cos = rotary_embedding(depth, seq_len)

    y_transformer = transformer_block(x, sin, cos, training=True)
    print("TransformerBlock output shape:", y_transformer.shape)

if __name__ == "__main__":
    main()
