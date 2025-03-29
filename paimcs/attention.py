from tensorflow import float16, Tensor, reshape, transpose, shape, reduce_sum, einsum, expand_dims
from tensorflow.keras import layers
from .rope import rotary_embedding, apply_rotary_pos_emb
from .projection import ProjectionWithKernel
from .favor import FAVORProjection
from .nystrom import NystromFeatures

class MultiHeadFAVORAttention(layers.Layer):
    def __init__(self, num_heads: int, attn_dim: int, num_random_features: int,
                 conv_channels: int, groups: int, num_features_per_scale: int, gamma_list: list,
                 nystrom_landmarks: int, dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Multi-head FAVOR+ attention with integrated Nyström approximation.
        Args:
            num_heads: Number of attention heads.
            attn_dim: Total attention dimension.
            num_random_features: Random features per head.
            conv_channels: Input channels for projection.
            groups: Groups for convolution.
            num_features_per_scale: Features per scale.
            gamma_list: List of gamma values.
            nystrom_landmarks: Number of landmarks for Nyström.
            dropout_rate: Dropout rate.
        """
        super().__init__(**kwargs, dtype=float16)
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // num_heads
        self.num_random_features = num_random_features
        self.dropout_rate = dropout_rate

        self.query_proj = ProjectionWithKernel(attn_dim, conv_channels, groups, num_features_per_scale, gamma_list, dropout_rate)
        self.key_proj   = ProjectionWithKernel(attn_dim, conv_channels, groups, num_features_per_scale, gamma_list, dropout_rate)
        self.value_proj = ProjectionWithKernel(attn_dim, conv_channels, groups, num_features_per_scale, gamma_list, dropout_rate)
        self.out_dense = layers.Dense(attn_dim)
        self.favor_proj = FAVORProjection(self.head_dim, self.num_random_features)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.nystrom = NystromFeatures(nystrom_landmarks, gamma=1.0)

    def split_heads(self, x: Tensor, batch_size: Tensor, seq_len: Tensor) -> Tensor:
        x = reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return transpose(x, perm=[0, 2, 1, 3])  # (batch, num_heads, seq_len, head_dim)

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        batch_size = shape(x)[0]
        seq_len = shape(x)[1]
        # Compute Q, K, V.
        q = self.query_proj(x, training=training)  # (batch, seq_len, attn_dim)
        k = self.key_proj(x, training=training)
        v = self.value_proj(x, training=training)
        # Split into heads.
        q = self.split_heads(q, batch_size, seq_len)
        k = self.split_heads(k, batch_size, seq_len)
        v = self.split_heads(v, batch_size, seq_len)
        # Apply rotary positional encoding.
        sin, cos = rotary_embedding(self.head_dim, seq_len)
        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k, sin, cos)
        # Apply Nyström approximation on keys.
        k = self.nystrom(k)
        # Apply FAVOR projection.
        q_prime = self.favor_proj(q) # (batch, num_heads, seq_len, num_random_features)
        k_prime = self.favor_proj(k) # (batch, num_heads, seq_len, num_random_features)
        # Normalization factor.
        k_prime_sum = reduce_sum(k_prime, axis=2)  # (batch, num_heads, num_random_features)
        denom = einsum('bhse,bhe->bhs', q_prime, k_prime_sum)
        denom = expand_dims(denom, -1)
        # Numerator.
        kv = einsum('bhse,bhsd->bhed', k_prime, v)
        numerator = einsum('bhse,bhed->bhsd', q_prime, kv)
        output = numerator / (denom + 1e-6)
        # Combine heads.
        output = transpose(output, perm=[0, 2, 1, 3])
        output = reshape(output, (batch_size, seq_len, self.attn_dim))
        output = self.out_dense(output)
        return self.dropout(output, training=training)