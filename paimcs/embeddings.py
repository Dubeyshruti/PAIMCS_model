from tensorflow.keras import layers
from .conv import GroupedPointwiseConv1D
from .multiscale import MultiScaleKernelFeatures
from tensorflow import float16, Tensor

class TokenRepresentation(layers.Layer):
    def __init__(self, vocab_size: int, hidden_dim: int, conv_output_channels: int, groups: int,
                 num_features_per_scale: int, gamma_list: list, dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Converts token IDs into rich continuous representations.
        Args:
            vocab_size: Vocabulary size.
            hidden_dim: Dimension of embeddings.
            conv_output_channels: Channels after convolution.
            groups: Number of groups for conv.
            num_features_per_scale: Features per scale.
            gamma_list: List of gamma values.
            dropout_rate: Dropout rate.
        """
        super().__init__(dtype = float16, **kwargs)
        self.embedding = layers.Embedding(vocab_size, hidden_dim)
        self.grouped_conv = GroupedPointwiseConv1D(hidden_dim, conv_output_channels, groups, dropout_rate)
        self.multi_scale_kernel = MultiScaleKernelFeatures(conv_output_channels, num_features_per_scale, gamma_list, dropout_rate)

    def call(self, token_seq: Tensor, training: bool = False) -> Tensor:
        x = self.embedding(token_seq)  # (batch, seq_len, hidden_dim)
        x = self.grouped_conv(x, training=training)  # (batch, seq_len, conv_output_channels)
        x = self.multi_scale_kernel(x, training=training)  # (batch, seq_len, num_features_per_scale*len(gamma_list))
        return x