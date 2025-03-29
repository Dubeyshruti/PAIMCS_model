# ==========================
# RMSNorm (using RMS normalization)
# ==========================
class RMSNorm(layers.Layer):
    def __init__(self, epsilon: float = 8.5e-6, **kwargs) -> None:
        """
        RMSNorm: Normalizes inputs based on their root-mean-square.
        Args:
            epsilon (float): Small constant for numerical stability.
        """
        super().__init__(**kwargs)
        self.epsilon: float = epsilon

    def build(self, input_shape: tf.TensorShape) -> None:
        if input_shape.rank is None or input_shape[-1] is None:
            raise ValueError("The last dimension must be defined.")
        self.gamma = self.add_weight(
            name="gamma",
            shape=input_shape[-1:],
            initializer="ones",
            trainable=True,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * self.gamma

# =================================================
# KernelLLM Block with RMSNorm and Multi-Head FAVOR+ Attention
# =================================================
class KernelLLMBlock(layers.Layer):
    def __init__(self, attn_dim: int, num_heads: int, num_random_features: int,
                 conv_channels: int, groups: int, num_features_per_scale: int, gamma_list: List[float],
                 nystrom_landmarks: int, dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Transformer-style block using multi-head FAVOR+ attention and RMSNorm.
        """
        super().__init__(**kwargs)
        self.attn = MultiHeadFAVORAttention(
            num_heads, attn_dim, num_random_features,
            conv_channels, groups, num_features_per_scale, gamma_list,
            nystrom_landmarks, dropout_rate
        )
        self.norm1 = RMSNorm()
        self.ff = tf.keras.Sequential([
            layers.Dense(attn_dim * 4, activation='relu'),
            layers.Dense(attn_dim)
        ])
        self.norm2 = RMSNorm()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_out = self.attn(x, training=training)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out, training=training))
        return x