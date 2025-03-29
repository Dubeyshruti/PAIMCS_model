# =====================
# Token Representation Module
# =====================
class TokenRepresentation(layers.Layer):
    def __init__(self, vocab_size: int, embedding_dim: int, conv_output_channels: int, groups: int,
                 num_features_per_scale: int, gamma_list: List[float], dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Converts token IDs into rich continuous representations.
        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Dimension of embeddings.
            conv_output_channels: Output channels after conv.
            groups: Groups for conv.
            num_features_per_scale: Features per scale.
            gamma_list: List of gamma values.
            dropout_rate: Dropout rate.
        """
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.grouped_conv = GroupedPointwiseConv1D(embedding_dim, conv_output_channels, groups, dropout_rate)
        self.multi_scale_kernel = MultiScaleKernelFeatures(conv_output_channels, num_features_per_scale, gamma_list, dropout_rate)

    def call(self, token_seq: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embedding(token_seq)  # (batch, seq_len, embedding_dim)
        x = self.grouped_conv(x, training=training)  # (batch, seq_len, conv_output_channels)
        x = self.multi_scale_kernel(x, training=training)  # (batch, seq_len, features)
        return x

# =============================================
# Full KernelLLM Model for Autoregressive Next-Token Prediction
# =============================================
class KernelLLM(Model):
    def __init__(self, vocab_size: int, max_seq_len: int, embedding_dim: int, conv_output_channels: int, groups: int,
                 num_features_per_scale: int, gamma_list: List[float], attn_dim: int, num_heads: int,
                 num_random_features: int, nystrom_landmarks: int, num_layers: int, dropout_rate: float = 0.1, **kwargs) -> None:
        """
        A modular large language model integrating token representation and multi-head FAVOR+ attention with Nyström.
        Args:
            vocab_size: Vocabulary size.
            max_seq_len: Maximum sequence length.
            embedding_dim: Token embedding dimension.
            conv_output_channels: Channels after grouped conv in token representation.
            groups: Groups for conv.
            num_features_per_scale: Features per scale.
            gamma_list: List of gamma values.
            attn_dim: Attention dimension.
            num_heads: Number of attention heads.
            num_random_features: Random features per head.
            nystrom_landmarks: Number of landmarks for Nyström.
            num_layers: Number of transformer blocks.
            dropout_rate: Dropout rate.
        """
        super().__init__(**kwargs)
        self.token_repr = TokenRepresentation(vocab_size, embedding_dim, conv_output_channels, groups,
                                                num_features_per_scale, gamma_list, dropout_rate)
        # External sinusoidal positional encoding is applied here.
        self.pos_encoding = PositionalEncoding(d_model=conv_output_channels * len(gamma_list), max_len=max_seq_len)
        self.proj = layers.Dense(attn_dim)
        self.dropout = layers.Dropout(dropout_rate)
        self.blocks = [KernelLLMBlock(attn_dim, num_heads, num_random_features,
                                      conv_output_channels, groups, num_features_per_scale, gamma_list,
                                      nystrom_landmarks, dropout_rate)
                       for _ in range(num_layers)]
        self.out_proj = layers.Dense(vocab_size)

    def call(self, token_seq: tf.Tensor, training: bool = False) -> tf.Tensor:
        # token_seq: (batch, seq_len)
        x = self.token_repr(token_seq, training=training)
        x = self.pos_encoding(x)
        x = self.proj(x)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        # For autoregressive next-token prediction, use the last token's representation.
        last_token = x[:, -1, :]
        logits = self.out_proj(last_token)
        return logits