import math
import tensorflow as tf

class OrthogonalRandomFeaturesTF(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, num_features: int, gamma: float = 1.0,
                 dropout_rate: float = 0.0, **kwargs) -> None:
        """
        Approximates an RBF kernel using orthogonal random features.
        Args:
            input_dim (int): Input dimension.
            num_features (int): Number of random features.
            gamma (float): RBF kernel parameter.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(dtype=tf.float16, **kwargs)
        self.input_dim = input_dim
        self.num_features = num_features
        self.gamma = gamma
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape) -> None:
        W = tf.random.normal((self.input_dim, self.num_features))
        q, _ = tf.linalg.qr(W)
        self.W = self.add_weight(
            "W", initializer=tf.constant_initializer(q * math.sqrt(2 * self.gamma)),
            trainable=False, dtype=tf.float16
        )
        self.b = self.add_weight(
            "b", shape=(self.num_features,),
            initializer=tf.random_uniform_initializer(0, 2 * math.pi),
            trainable=False, dtype = tf.float16
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        projection = tf.tensordot(x, self.W, axes=1) + self.b
        rff = math.sqrt(2.0 / self.num_features) * tf.math.cos(projection)
        return self.dropout(rff, training=training)