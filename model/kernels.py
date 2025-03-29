# ==============================
# Nyström Features (for approximating keys)
# ==============================
class NystromFeatures(layers.Layer):
    def __init__(self, num_landmarks: int, gamma: float = 1.0, **kwargs) -> None:
        """
        Nyström layer to approximate an RBF kernel.
        Args:
            num_landmarks (int): Number of landmark points.
            gamma (float): RBF kernel parameter.
        """
        super().__init__(**kwargs)
        self.num_landmarks: int = num_landmarks
        self.gamma: float = gamma

    def build(self, input_shape: tf.TensorShape) -> None:
        self.input_dim = input_shape[-1]
        self.landmarks = self.add_weight(
            shape=(self.num_landmarks, self.input_dim),
            initializer="random_normal",
            trainable=True,
            name="landmarks"
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (batch, seq_len, d)
        x2 = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.reduce_sum(tf.square(self.landmarks), axis=-1, keepdims=True)
        l2 = tf.transpose(l2)
        dot = tf.matmul(inputs, self.landmarks, transpose_b=True)
        dist_sq = x2 + l2 - 2.0 * dot
        features = tf.exp(-self.gamma * dist_sq)
        return features

# ==========================================
# Orthogonal Random Features for Multi-Scale Kernel Features
# ==========================================
class OrthogonalRandomFeaturesTF(layers.Layer):
    def __init__(self, input_dim: int, num_features: int, gamma: float = 1.0, dropout_rate: float = 0.0, **kwargs) -> None:
        """
        Approximates an RBF kernel using orthogonal random features.
        Args:
            input_dim (int): Input dimension.
            num_features (int): Number of random features.
            gamma (float): RBF kernel parameter.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_features = num_features
        self.gamma = gamma
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape) -> None:
        W = tf.random.normal((self.input_dim, self.num_features))
        q, _ = tf.linalg.qr(W)
        self.W = self.add_weight(
            "W", initializer=tf.constant_initializer(q * math.sqrt(2 * self.gamma)),
            trainable=False
        )
        self.b = self.add_weight(
            "b", shape=(self.num_features,),
            initializer=tf.random_uniform_initializer(0, 2 * math.pi),
            trainable=False
        )
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        projection = tf.tensordot(x, self.W, axes=1) + self.b
        rff = math.sqrt(2.0 / self.num_features) * tf.math.cos(projection)
        return self.dropout(rff, training=training)

# =========================
# Multi-Scale Kernel Features Module
# =========================
class MultiScaleKernelFeatures(layers.Layer):
    def __init__(self, input_dim: int, num_features_per_scale: int, gamma_list: List[float], dropout_rate: float = 0.0, **kwargs) -> None:
        """
        Concatenates random Fourier features computed at multiple scales.
        Args:
            input_dim (int): Input dimension.
            num_features_per_scale (int): Features per scale.
            gamma_list (List[float]): List of gamma values.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(**kwargs)
        self.scales = [OrthogonalRandomFeaturesTF(input_dim, num_features_per_scale, gamma, dropout_rate)
                       for gamma in gamma_list]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        features = [scale(x, training=training) for scale in self.scales]
        return tf.concat(features, axis=-1)

# ========================================
# Grouped Pointwise 1D Convolution for Feature Extraction
# ========================================
class GroupedPointwiseConv1D(layers.Layer):
    def __init__(self, input_channels: int, output_channels: int, groups: int = 1, dropout_rate: float = 0.1, **kwargs) -> None:
        """
        Grouped 1D pointwise convolution.
        Args:
            input_channels (int): Input channel dimension.
            output_channels (int): Output channel dimension.
            groups (int): Number of groups.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(**kwargs)
        if input_channels % groups != 0 or output_channels % groups != 0:
            raise ValueError("Channels must be divisible by groups.")
        self.conv = layers.Conv1D(filters=output_channels, kernel_size=1, groups=groups)
        self.activation = layers.ReLU()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return self.dropout(x, training=training)