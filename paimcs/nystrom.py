from tensorflow.keras import layers
from tensorflow import float16, TensorShape, Tensor, reduce_sum, square, transpose, matmul, exp

class NystromFeatures(layers.Layer):
    def __init__(self, num_landmarks: int, gamma: float = 1.0, **kwargs) -> None:
        """
        Nyström layer to approximate an RBF kernel.
        Args:
            num_landmarks (int): Number of landmark points.
            gamma (float): RBF kernel parameter.
        """
        super().__init__(**kwargs, dtype=float16)
        self.num_landmarks = num_landmarks
        self.gamma = gamma

    def build(self, input_shape: TensorShape) -> None:
        self.input_dim = input_shape[-1]
        self.landmarks = self.add_weight(
            shape=(self.num_landmarks, self.input_dim),
            initializer="random_normal",
            trainable=True,
            name="landmarks"
        )
        super().build(input_shape, dtype = float16)

    def call(self, inputs: Tensor) -> Tensor:
        # Compute squared L2 distances between inputs and landmarks
        x2 = reduce_sum(square(inputs), axis=-1, keepdims=True)
        l2 = reduce_sum(square(self.landmarks), axis=-1, keepdims=True)
        l2 = transpose(l2)
        dot = matmul(inputs, self.landmarks, transpose_b=True)
        dist_sq = x2 + l2 - 2.0 * dot
        features = exp(-self.gamma * dist_sq)
        return features