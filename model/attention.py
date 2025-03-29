# ===============================
# ProjectionWithKernel for Q/K/V Projections
# ===============================
class ProjectionWithKernel(layers.Layer):
    def __init__(self, output_dim: int, conv_channels: int, groups: int,
                 num_features_per_scale: int, gamma_list: List[float], dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Projects inputs to a desired dimension using grouped conv and multi-scale kernel features.
        Args:
            output_dim: Target output dimension.
            conv_channels: Input channels for conv.
            groups: Number of groups.
            num_features_per_scale: Features per scale.
            gamma_list: List of gamma values.
            dropout_rate: Dropout probability.
        """
        super().__init__(**kwargs)
        self.conv_proj = GroupedPointwiseConv1D(conv_channels, conv_channels, groups, dropout_rate)
        self.multi_scale = MultiScaleKernelFeatures(conv_channels, num_features_per_scale, gamma_list, dropout_rate)
        self.final_dense = layers.Dense(output_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv_proj(x, training=training)
        x = self.multi_scale(x, training=training)
        return self.final_dense(x)

# ========================================
# Multi-Head FAVOR+ Attention with Nyström Integration
# ========================================
class MultiHeadFAVORAttention(layers.Layer):
    def __init__(self, num_heads: int, attn_dim: int, num_random_features: int,
                 conv_channels: int, groups: int, num_features_per_scale: int, gamma_list: List[float],
                 nystrom_landmarks: int, dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Multi-head FAVOR+ attention using ProjectionWithKernel for Q/K/V and fully integrated Nyström for keys.
        Args:
            num_heads: Number of attention heads.
            attn_dim: Total attention dimension.
            num_random_features: Random features per head.
            conv_channels: Input channels for projection.
            groups: Groups for conv.
            num_features_per_scale: Features per scale for projection.
            gamma_list: List of gamma values.
            nystrom_landmarks: Number of landmarks for Nyström.
            dropout_rate: Dropout rate.
        """
        super().__init__(**kwargs)
        self.num_heads: int = num_heads
        self.attn_dim: int = attn_dim
        self.head_dim: int = attn_dim // num_heads
        self.num_random_features: int = num_random_features
        self.dropout_rate: float = dropout_rate

        self.query_proj = ProjectionWithKernel(attn_dim, conv_channels, groups, num_features_per_scale, gamma_list, dropout_rate)
        self.key_proj   = ProjectionWithKernel(attn_dim, conv_channels, groups, num_features_per_scale, gamma_list, dropout_rate)
        self.value_proj = ProjectionWithKernel(attn_dim, conv_channels, groups, num_features_per_scale, gamma_list, dropout_rate)
        self.out_dense = layers.Dense(attn_dim)
        self.favor_proj = FAVORProjection(self.head_dim, self.num_random_features)
        self.dropout = layers.Dropout(self.dropout_rate)
        # Fully integrate Nyström for keys (non-optional)
        self.nystrom = NystromFeatures(nystrom_landmarks, gamma=1.0)

    def split_heads(self, x: tf.Tensor, batch_size: tf.Tensor, seq_len: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        # Compute Q, K, V using kernel-based projection.
        q = self.query_proj(x, training=training)  # (batch, seq_len, attn_dim)
        k = self.key_proj(x, training=training)
        v = self.value_proj(x, training=training)
        # Split heads.
        q = self.split_heads(q, batch_size, seq_len)  # (batch, num_heads, seq_len, head_dim)
        k = self.split_heads(k, batch_size, seq_len)
        v = self.split_heads(v, batch_size, seq_len)
        # Apply rotary positional encoding.
        sin, cos = rotary_embedding(self.head_dim, seq_len)
        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k, sin, cos)
        # Apply Nyström approximation on keys.
        k = self.nystrom(k)  # Now k: (batch, num_heads, seq_len, nystrom_landmarks)
        # Apply FAVOR projection.
        q_prime = self.favor_proj(q)  # (batch, num_heads, seq_len, num_random_features)
        k_prime = self.favor_proj(k)  # (batch, num_heads, seq_len, num_random_features)
        # Normalization factor.
        k_prime_sum = tf.reduce_sum(k_prime, axis=2)  # (batch, num_heads, num_random_features)
        denom = tf.einsum('bhse,bhe->bhs', q_prime, k_prime_sum)  # (batch, num_heads, seq_len)
        denom = tf.expand_dims(denom, -1)  # (batch, num_heads, seq_len, 1)
        # Numerator.
        kv = tf.einsum('bhse,bhsd->bhed', k_prime, v)  # (batch, num_heads, num_random_features, head_dim)
        numerator = tf.einsum('bhse,bhed->bhsd', q_prime, kv)  # (batch, num_heads, seq_len, head_dim)
        output = numerator / (denom + 1e-6)
        # Combine heads.
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch, seq_len, num_heads, head_dim)
        output = tf.reshape(output, (batch_size, seq_len, self.attn_dim))
        output = self.out_dense(output)
        return self.dropout(output, training=training)