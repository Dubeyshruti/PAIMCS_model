from tensorflow.keras import layers
from tensorflow import float16, TensorShape, random_normal_initializer, random_uniform_initializer, Tensor, tensordot, reduce_sum, square, exp
from math import pi

class FAVORProjection(layers.Layer):
    def __init__(self, input_dim: int, num_features: int, **kwargs) -> None:
        """
        Computes the random feature map φ(x) for approximating softmax attention using FAVOR+.
        Args:
            input_dim (int): Dimensionality of input vectors.
            num_features (int): Number of random features.
        """
        super().__init__(**kwargs, dtype=float16)
        self.input_dim = input_dim
        self.num_features = num_features

    def build(self, input_shape: TensorShape) -> None:
        self.W = self.add_weight(
            shape=(self.input_dim, self.num_features),
            initializer=random_normal_initializer(),
            dtype = float16
            trainable=False,
            name="favor_W"
        )
        self.bias = self.add_weight(
            shape=(self.num_features,),
            initializer=random_uniform_initializer(0, 2 * pi),
            dtype = float16
            trainable=False,
            name="favor_bias"
        )
        super().build(input_shape, dtype = float16)

    def call(self, x: Tensor) -> Tensor:
        x_proj = tensordot(x, self.W, axes=1) + self.bias
        norm_sq = reduce_sum(square(x), axis=-1, keepdims=True)
        return exp(-0.5 * norm_sq) * exp(x_proj)