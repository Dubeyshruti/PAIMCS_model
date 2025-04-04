import math
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# =============================================================================
# Custom Initializer for Orthogonal Random Features
# =============================================================================
def get_orthogonal_matrix(input_dim: int, num_features: int, gamma: float):
    """
    Generates an orthogonal matrix of the specified dimensions using QR decomposition.
    Args:
        input_dim (int): The number of rows (input dimension).
        num_features (int): The number of columns (number of features).
    Returns:
        numpy.ndarray: An orthogonal matrix of shape (input_dim, num_features).
                       Note: If input_dim < num_features, the resulting matrix will have orthogonal columns.
                             If input_dim > num_features, the resulting matrix will have orthogonal rows (after transpose).
                             If input_dim == num_features, it will be a square orthogonal matrix.
    """
    # Generate a random matrix
    random_matrix = np.random.rand(input_dim, num_features)

    # Perform QR decomposition
    q, r = np.linalg.qr(random_matrix)
    scale = math.sqrt(2 * gamma)
    q_scaled = q * scale

    return q_scaled.astype(np.float16) # Ensure the data type is compatible with TensorFlow

# =============================================================================
# 1. RMSNorm (Root Mean Square Normalization)
# =============================================================================
class RMSNorm(layers.Layer):
    def __init__(self, epsilon: float = 8.5e-6, **kwargs) -> None:
        """
        RMSNorm normalizes inputs using their root-mean-square.
        Args:
            epsilon (float): Small constant to avoid division by zero.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.epsilon: float = epsilon

    def build(self, input_shape: tf.TensorShape) -> None:
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank is None or input_shape[-1] is None:
            raise ValueError("The last dimension must be defined.")
        self.gamma = self.add_weight(
            name="gamma",
            shape=input_shape[-1:],
            initializer="ones",
            trainable=True,
            dtype=tf.float16
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * self.gamma

# =============================================================================
# 2. Absolute Positional Encoding (Sinusoidal)
# =============================================================================
class PositionalEncoding(layers.Layer):
    def __init__(self, hidden_dim: int, max_len: int = 678, **kwargs) -> None:
        """
        Computes sinusoidal positional encodings.
        Args:
            hidden_dim (int): Dimension of token representations.
            max_len (int): Maximum sequence length.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.hidden_dim: int = hidden_dim
        pos_encoding = self.get_positional_encoding(max_len, hidden_dim)
        self.pos_encoding: tf.Tensor = tf.cast(pos_encoding, tf.float16)

    def get_positional_encoding(self, max_len: int, hidden_dim: int) -> tf.Tensor:
        pos = tf.cast(tf.range(max_len)[:, tf.newaxis], dtype=tf.float16)  # (max_len, 1)
        i = tf.cast(tf.range(hidden_dim)[tf.newaxis, :], dtype = tf.float16)  # (1, hidden_dim)
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(hidden_dim, tf.float16))
        angle_rads = pos * angle_rates  # (max_len, hidden_dim)
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, max_len, hidden_dim)
        return pos_encoding

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

# =============================================================================
# 3. Rotary Positional Encoding Functions (for relative positions)
# =============================================================================
def rotary_embedding(head_dim: int, seq_len: int, base: int = 12315) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes rotary positional embeddings (sine and cosine).
    Args:
        head_dim (int): Must be even.
        seq_len (int): Sequence length.
        base (int): Frequency base.
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Sine and cosine embeddings of shape (1, seq_len, head_dim//2)
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even.")
    pos = tf.cast(tf.range(seq_len), tf.float16)
    dim_range = tf.cast(tf.range(0, head_dim, 2), tf.float16)
    inv_freq = 1.0 / tf.pow(tf.cast(base, tf.float16), dim_range / head_dim)
    sinusoid_inp = tf.expand_dims(pos, -1) * tf.expand_dims(inv_freq, 0)  # (seq_len, head_dim//2)
    sine = tf.sin(sinusoid_inp)[tf.newaxis, ...]   # (1, seq_len, head_dim//2)
    cosine = tf.cos(sinusoid_inp)[tf.newaxis, ...]  # (1, seq_len, head_dim//2)
    return sine, cosine

def apply_rotary_pos_emb(x: tf.Tensor, sin: tf.Tensor, cos: tf.Tensor) -> tf.Tensor:
    """
    Applies rotary positional embeddings.
    Args:
        x (tf.Tensor): Shape (batch, num_heads, seq_len, head_dim)
        sin, cos (tf.Tensor): Shapes (1, seq_len, head_dim//2)
    Returns:
        tf.Tensor: Tensor with rotary encoding applied.
    """
    d_half = tf.shape(x)[-1] // 2
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    sin = tf.cast(sin, x.dtype)
    cos = tf.cast(cos, x.dtype)
    sin = tf.reshape(sin, (1, 1, tf.shape(sin)[1], d_half))
    cos = tf.reshape(cos, (1, 1, tf.shape(cos)[1], d_half))
    return tf.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

# =============================================================================
# 4. FAVORProjection Layer (for FAVOR+ attention)
# =============================================================================
class FAVORProjection(layers.Layer):
    def __init__(self, input_dim: int, num_features: int, **kwargs) -> None:
        """
        Computes the random feature map φ(x) for approximating softmax attention using FAVOR+.
        Args:
            input_dim (int): Dimension of the input (should equal head_dim).
            num_features (int): Number of random features.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.input_dim: int = input_dim
        self.num_features: int = num_features

    def build(self, input_shape: tf.TensorShape) -> None:
        self.W = self.add_weight(
            shape=(self.input_dim, self.num_features),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            dtype=tf.float16,
            name="favor_W"
        )
        self.bias = self.add_weight(
            shape=(self.num_features,),
            initializer=tf.random_uniform_initializer(0, 2 * math.pi),
            trainable=False,
            dtype=tf.float16,
            name="favor_bias"
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_proj = tf.matmul(x, self.W) + self.bias
        norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        return tf.exp(-0.5 * norm_sq) * tf.exp(x_proj)

# =============================================================================
# 5. Orthogonal Random Features for Multi-Scale Kernel Features
# =============================================================================
class OrthogonalRandomFeaturesTF(layers.Layer):
    def __init__(self, input_dim: int, num_features: int, gamma: float, dropout_rate: float, **kwargs) -> None:
        """
        Approximates an RBF kernel using orthogonal random features.
        Args:
            input_dim (int): Input dimension.
            num_features (int): Number of random features.
            gamma (float): RBF kernel parameter.
            dropout_rate (float): Dropout probability.
            precomputed_orthogonal_matrix (np.ndarray): A pre-computed orthogonal matrix of shape (input_dim, num_features).
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.input_dim: int = input_dim
        self.num_features: int = num_features
        self.gamma: float = gamma
        self.dropout_rate: float = dropout_rate
        self.precomputed_orthogonal_matrix = get_orthogonal_matrix(input_dim, num_features, gamma)

    def build(self, input_shape: tf.TensorShape) -> None:
        initializer = tf.constant(self.precomputed_orthogonal_matrix, dtype = self.dtype)
        self.W = self.add_weight(
           name= "W",
           shape=(self.input_dim, self.num_features),
           initializer = self.precomputed_orthogonal_matrix,
           trainable=False,
           dtype=tf.float16
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.num_features,),
            initializer=tf.random_uniform_initializer(0, 2 * math.pi),
            trainable=False,
            dtype=tf.float16
        )
        self.dropout = layers.Dropout(self.dropout_rate, dtype=tf.float16)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        projection = tf.matmul(x, self.W) + self.b
        rff = math.sqrt(2.0 / self.num_features) * tf.math.cos(projection)
        return self.dropout(rff, training=training)

# =============================================================================
# 6. Multi-Scale Kernel Features Module
# =============================================================================
class MultiScaleKernelFeatures(layers.Layer):
    def __init__(self, input_dim: int, num_features_per_scale: int, gamma_list: List[float], dropout_rate: float, **kwargs) -> None:
        """
        Concatenates random Fourier features computed at multiple scales.
        Args:
            input_dim (int): Input dimension.
            num_features_per_scale (int): Features per scale.
            gamma_list (List[float]): List of gamma values.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.scales = [OrthogonalRandomFeaturesTF(input_dim, num_features_per_scale, gamma, dropout_rate)
                       for gamma in gamma_list]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        features = [scale(x, training=training) for scale in self.scales]
        return tf.concat(features, axis=-1)

# =============================================================================
# 7. Grouped Pointwise 1D Convolution for Feature Extraction
# =============================================================================
class GroupedPointwiseConv1D(layers.Layer):
    def __init__(self, input_channels: int, output_channels: int, groups: int, dropout_rate: float, **kwargs) -> None:
        """
        Grouped 1D pointwise convolution.
        Args:
            input_channels (int): Input channel dimension.
            output_channels (int): Output channel dimension.
            groups (int): Number of groups.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        if input_channels % groups != 0 or output_channels % groups != 0:
            raise ValueError("Channels must be divisible by groups.")
        self.conv = layers.Conv1D(filters=output_channels, kernel_size=1, groups=groups, dtype=tf.float16)
        self.activation = lambda x: x * tf.sigmoid(x)
        self.dropout = layers.Dropout(dropout_rate, dtype = tf.float16)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return self.dropout(x, training=training)

# =============================================================================
# 8. Token Representation Module
# =============================================================================
class TokenRepresentation(layers.Layer):
    def __init__(self, vocab_size: int=31542, token_input_dim: int=339, groups: int=3, max_seq_len: int=678, num_features_per_scale: int=120,
                 gamma_list: List[float]=[99e-5, 0.099, 9.9], dropout_rate: float = 0.099, **kwargs) -> None:
        """
        Converts token IDs into rich continuous representations.
        Flow:
          1. Embed tokens.
          2. Apply grouped pointwise convolution.
          3. Apply absolute positional encoding.
          4. Map features using multi-scale kernel features.
        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Dimension of raw token embeddings.
            conv_output_channels (int): Output channels from convolution.
            groups (int): Number of groups for convolution.
            max_seq_len (int): Maximum sequence length (for positional encoding).
            num_features_per_scale (int): Features per scale.
            gamma_list (List[float]): List of gamma values.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.embedding = layers.Embedding(vocab_size, token_input_dim, dtype=tf.float16)
        self.pos_encoding = PositionalEncoding(hidden_dim=token_input_dim, max_len=max_seq_len)
        self.multi_scale_kernel = MultiScaleKernelFeatures(token_input_dim, num_features_per_scale, gamma_list, dropout_rate)

    def call(self, token_seq: tf.Tensor, training: bool = False) -> tf.Tensor:
        # token_seq: (batch, seq_len)
        x = self.embedding(token_seq)                              # (batch, seq_len, embedding_dim)
        x = self.pos_encoding(x)                                     # (batch, seq_len, conv_output_channels)
        x = self.multi_scale_kernel(x, training=training)            # (batch, seq_len, num_features_per_scale * len(gamma_list))
        return x

# =============================================================================
# 9. Multi-Head FAVOR+ Attention with Alternating Projections and Nyström Integration
# =============================================================================
class MultiHeadFAVORAttention(layers.Layer):
    def __init__(self, num_heads: int=3, attn_dim: int=360, num_random_features: int=87, groups: int=3, num_features_per_scale: int=120,
                 gamma_list: List[float]=[99e-5, 0.099, 9.9], proj_type: str='kernel', dropout_rate: float = 0.099, **kwargs) -> None:
        """
        Multi-head FAVOR+ attention using alternating projection mechanisms for Q/K/V.
        Args:
            num_heads (int): Number of attention heads.
            attn_dim (int): Total attention dimension.
            num_random_features (int): Random features per head.
            conv_channels (int): Input channels for projection.
            groups (int): Number of groups for convolution.
            num_features_per_scale (int): Features per scale for projection.
            gamma_list (List[float]): List of gamma values.
            nystrom_landmarks (int): Number of landmarks for Nyström.
            proj_type (str): "conv" or "kernel" – the projection mechanism to use.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // num_heads
        self.num_random_features = num_random_features
        self.dropout_rate = dropout_rate

        if proj_type == "conv":
            self.query_proj = GroupedPointwiseConv1D(attn_dim, attn_dim, groups, dropout_rate)
            self.key_proj = GroupedPointwiseConv1D(attn_dim, attn_dim, groups, dropout_rate)
            self.value_proj = GroupedPointwiseConv1D(attn_dim, attn_dim, groups, dropout_rate)
        elif proj_type == "kernel":
            self.query_proj = MultiScaleKernelFeatures(attn_dim, num_features_per_scale, gamma_list, dropout_rate)
            self.key_proj = MultiScaleKernelFeatures(attn_dim, num_features_per_scale, gamma_list, dropout_rate)
            self.value_proj = MultiScaleKernelFeatures(attn_dim, num_features_per_scale, gamma_list, dropout_rate)
        else:
            raise ValueError("proj_type must be either 'conv' or 'kernel'")

        self.out_dense = layers.Dense(attn_dim, dtype=tf.float16)
        self.favor_proj = FAVORProjection(self.head_dim, self.num_random_features)
        self.dropout = layers.Dropout(self.dropout_rate, dtype=tf.float16)

    def split_heads(self, x: tf.Tensor, batch_size: tf.Tensor, seq_len: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, num_heads, seq_len, head_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        # Compute Q, K, V projections.
        q = self.query_proj(x, training=training)  # (batch, seq_len, attn_dim)
        k = self.key_proj(x, training=training)
        v = self.value_proj(x, training=training)
        # Split into heads.
        q = self.split_heads(q, batch_size, seq_len)  # (batch, num_heads, seq_len, head_dim)
        k = self.split_heads(k, batch_size, seq_len)
        v = self.split_heads(v, batch_size, seq_len)
        # Apply rotary positional encoding.
        sin, cos = rotary_embedding(self.head_dim, seq_len)
        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k, sin, cos)
        # Apply FAVOR projection.
        q_prime = self.favor_proj(q)  # (batch, num_heads, seq_len, num_random_features)
        k_prime = self.favor_proj(k)  # (batch, num_heads, seq_len, num_random_features)
        # Compute normalization factor.
        k_prime_sum = tf.reduce_sum(k_prime, axis=2, keepdims=True)  # (batch, num_heads, num_random_features)
        denom = tf.matmul(q_prime, tf.transpose(k_prime_sum, perm=[0, 1, 3, 2])) # (batch, num_heads, seq_len, 1)
        # Compute numerator.
        kv = tf.matmul(tf.transpose(k_prime, perm=[0, 1, 3, 2]), v)  # (batch, num_heads, num_random_features, head_dim)
        numerator = tf.matmul(q_prime, kv)  # (batch, num_heads, seq_len, head_dim)
        output = numerator / (denom + 1e-6)
        # Combine heads.
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch, seq_len, num_heads, head_dim)
        output = tf.reshape(output, (batch_size, seq_len, self.attn_dim))
        output = self.out_dense(output)
        return self.dropout(output, training=training)

# =============================================================================
# 11. KernelLMBlock (Transformer-style block with FAVOR+ Attention and RMSNorm)
# =============================================================================
class KernelLMBlock(layers.Layer):
    def __init__(self, attn_dim: int=360, num_heads: int=3, num_random_features: int=87, groups: int=3, num_features_per_scale: int=120,
                 gamma_list: List[float]=[99e-5, 0.099, 9.9], dropout_rate: float = 0.099, proj_type: str = "kernel", **kwargs) -> None:
        """
        Transformer block using multi-head FAVOR+ attention and a feed-forward network with RMSNorm.
        Args:
            attn_dim (int): Hidden dimension for attention.
            num_heads (int): Number of attention heads.
            num_random_features (int): Random features per head.
            conv_channels (int): Channels for projection in token representation.
            groups (int): Number of groups for convolution.
            num_features_per_scale (int): Features per scale.
            gamma_list (List[float]): List of gamma values.
            nystrom_landmarks (int): Number of landmarks for Nyström.
            dropout_rate (float): Dropout rate.
            proj_type (str): Projection type for Q/K/V ("conv" or "kernel").
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.attn = MultiHeadFAVORAttention(
            num_heads, attn_dim, num_random_features, groups, num_features_per_scale, gamma_list,
            proj_type, dropout_rate
        )
        self.norm1 = RMSNorm()
        self.ff = tf.keras.Sequential([
            layers.Dense(attn_dim * 3, activation=lambda x: x * tf.sigmoid(x), dtype=tf.float16),
            layers.Dense(attn_dim, dtype=tf.float16)
        ])
        self.norm2 = RMSNorm()
        self.dropout = layers.Dropout(dropout_rate, dtype=tf.float16)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_out = self.attn(x, training=training)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out, training=training))
        return x

# =============================================================================
# 13. paimcsLm Model (Autoregressive Next-Token Prediction Model)
# =============================================================================
class paimcsLm(Model):
    def __init__(self, vocab_size: int=31542, max_seq_len: int=678, embedding_dim: int=360, token_input_dim: int=339, groups: int=3,
                 num_features_per_scale: int=120, gamma_list: List[float]=[99e-5, 0.099, 9.9], num_heads: int=3,
                 num_random_features: int=87, num_layers: int=23, dropout_rate: float = 0.099, **kwargs) -> None:
        """
        A modular large language model for autoregressive next-token prediction.
        Flow:
          1. TokenRepresentation: Embedding -> Grouped Conv -> Absolute Positional Encoding -> MultiScale Kernel Mapping.
          2. Pass the resulting representation (already in the desired hidden dimension) to a stack of KernelLMBlocks.
          3. Output projection to vocabulary logits.
        Args:
            vocab_size (int): Vocabulary size.
            max_seq_len (int): Maximum sequence length.
            embedding_dim (int): Dimension of raw token embeddings.
            conv_output_channels (int): Output channels from convolution.
            groups (int): Number of groups for convolution.
            num_features_per_scale (int): Features per scale.
            gamma_list (List[float]): List of gamma values.
            attn_dim (int): Hidden (attention) dimension.
            num_heads (int): Number of attention heads.
            num_random_features (int): Random features per head.
            nystrom_landmarks (int): Number of landmarks for Nyström.
            num_layers (int): Number of transformer blocks.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.token_repr = TokenRepresentation(vocab_size, token_input_dim, groups, max_seq_len, num_features_per_scale, gamma_list, dropout_rate)
        # Assume TokenRepresentation outputs the desired hidden dimension.
        self.dropout = layers.Dropout(dropout_rate, dtype=tf.float16)
        # Alternate projection type across blocks.
        self.blocks = []
        for i in range(num_layers):
            proj_type = "kernel" if i % 3 == 0 else "conv"
            block = KernelLMBlock(
                embedding_dim, num_heads, num_random_features, groups, num_features_per_scale, gamma_list,
                dropout_rate, proj_type
            )
            self.blocks.append(block)
        self.out_proj = layers.Dense(vocab_size, dtype=tf.float16)

    def call(self, token_seq: tf.Tensor, training: bool = False) -> tf.Tensor:
        # token_seq: (batch, seq_len)
        x = self.token_repr(token_seq, training=training)  # (batch, seq_len, hidden_dim)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        # For next-token prediction, use the last token's representation.
        last_token = x[:, -1, :]
        logits = self.out_proj(last_token)  # (batch, vocab_size)
        return logits
