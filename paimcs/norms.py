from tensorflow import float16, TensorShape, Tensor, sqrt, reduce_mean, square
from tensorflow.keras import layers

class RMSNorm(layers.Layer):
    def __init__(self, epsilon: float = 8.5e-6, **kwargs) -> None:
        """
        RMSNorm: Normalizes inputs based on their root-mean-square.
        Args:
            epsilon (float): Small constant for numerical stability.
        """
        super().__init__(**kwargs, dtype=float16)
        self.epsilon = epsilon

    def build(self, input_shape: TensorShape) -> None:
        if input_shape.rank is None or input_shape[-1] is None:
            raise ValueError("The last dimension must be defined.")
        self.gamma = self.add_weight(
            name="gamma",
            shape=input_shape[-1:],
            initializer="ones",
            trainable=True,
            dtype=float16
        )
        super().build(input_shape)

    def call(self, x: Tensor) -> Tensor:
        rms = sqrt(reduce_mean(square(x), axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * self.gamma