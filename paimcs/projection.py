from tensorflow import Tensor
from tensorflow.keras import layers
from .conv import GroupedPointwiseConv1D
from .multiscale import MultiScaleKernelFeatures

class ProjectionWithKernel(layers.Layer):
    def __init__(self, output_dim: int, conv_channels: int, groups: int,
                 num_features_per_scale: int, gamma_list: list, dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Projects inputs to a desired dimension using grouped conv and multi-scale kernel features.
        Args:
            output_dim: Target output dimension.
            conv_channels: Input channels for conv.
            groups: Number of groups.
            num_features_per_scale: Features per scale.
            gamma_list: List of gamma values.
            dropout_rate: Dropout rate.
        """
        super().__init__(**kwargs)
        self.conv_proj = GroupedPointwiseConv1D(conv_channels, conv_channels, groups, dropout_rate)
        self.multi_scale = MultiScaleKernelFeatures(conv_channels, num_features_per_scale, gamma_list, dropout_rate)
        self.final_dense = layers.Dense(output_dim)

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        x = self.conv_proj(x, training=training)
        x = self.multi_scale(x, training=training)
        return self.final_dense(x)