from tensorflow.keras import layers
from .orthogonal import OrthogonalRandomFeaturesTF
from tensorflow import Tensor, concat, float16

class MultiScaleKernelFeatures(layers.Layer):
    def __init__(self, input_dim: int, num_features_per_scale: int, gamma_list: list,
                 dropout_rate: float = 0.0, **kwargs) -> None:
        """
        Concatenates random Fourier features computed at multiple scales.
        Args:
            input_dim (int): Input dimension.
            num_features_per_scale (int): Features per scale.
            gamma_list (List[float]): List of gamma values.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(**kwargs, dtype=float16)
        self.scales = [OrthogonalRandomFeaturesTF(input_dim, num_features_per_scale, gamma, dropout_rate)
                       for gamma in gamma_list]

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        features = [scale(x, training=training) for scale in self.scales]
        return concat(features, axis=-1)