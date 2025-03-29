from tensorflow import Tensor, float16
from tensorflow.keras import layers
from tensorflow.nn import silu

class GroupedPointwiseConv1D(layers.Layer):
    def __init__(self, input_channels: int, output_channels: int, groups: int = 1,
                 dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Grouped 1D pointwise convolution.
        Args:
            input_channels (int): Input channel dimension.
            output_channels (int): Output channel dimension.
            groups (int): Number of groups.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(dtype = float16, **kwargs)
        if input_channels % groups != 0 or output_channels % groups != 0:
            raise ValueError("Channels must be divisible by groups.")
        self.conv = layers.Conv1D(filters=output_channels, kernel_size=1, groups=groups)
        self.activation = silu
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return self.dropout(x, training=training)