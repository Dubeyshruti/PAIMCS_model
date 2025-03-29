from tensorflow import float16, Tensor
from tensorflow.keras import layers, Sequential
from .norms import RMSNorm
from .attention import MultiHeadFAVORAttention

class KernelLMBlock(layers.Layer):
    def __init__(self, attn_dim: int, num_heads: int, num_random_features: int,
                 conv_channels: int, groups: int, num_features_per_scale: int, gamma_list: list,
                 nystrom_landmarks: int, dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Transformer-style block using multi-head FAVOR+ attention and RMSNorm.
        """
        super().__init__(**kwargs, float16)
        self.attn = MultiHeadFAVORAttention(
            num_heads, attn_dim, num_random_features,
            conv_channels, groups, num_features_per_scale, gamma_list,
            nystrom_landmarks, dropout_rate
        )
        self.norm1 = RMSNorm()
        self.ff = Sequential([
            layers.Dense(attn_dim * 4, activation='silu'),
            layers.Dense(attn_dim)
        ])
        self.norm2 = RMSNorm()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        attn_out = self.attn(x, training=training)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out, training=training))
        return x