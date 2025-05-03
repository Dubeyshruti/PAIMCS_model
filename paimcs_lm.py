import tensorflow as tf

class DynamicConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_heads, channels, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.padding = kernel_size // 2

        # Sub-layer: predicts per-position, per-head kernels
        self.kernel_predict = tf.keras.layers.Dense(kernel_size * num_heads)

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            raise ValueError(
                f"DynamicConv1D expected input channels={self.channels}, "
                f"but got {input_shape[-1]}"
            )
        self.kernel_predict.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        kernels = self.kernel_predict(x)
        kernels = tf.reshape(kernels,
                             [batch, seq_len, self.num_heads, self.kernel_size])

        x_heads = tf.reshape(x,
                             [batch, seq_len, self.num_heads, self.head_dim])
        x_padded = tf.pad(
            x_heads,
            [[0, 0], [self.padding, self.padding], [0, 0], [0, 0]]
        )

        outputs = []
        for i in range(self.kernel_size):
            slice_i = x_padded[:, i:i + seq_len, :, :]
            weight_i = tf.expand_dims(kernels[:, :, :, i], -1)
            outputs.append(weight_i * slice_i)
        out = tf.add_n(outputs)
        out = tf.reshape(out, [batch, seq_len, self.channels])
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "num_heads": self.num_heads,
            "channels": self.channels
        })
        return config

class LocalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, heads, head_dim, window_size, **kwargs):
        super().__init__(**kwargs)
        assert window_size % 2 == 1, "window_size must be odd"
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size

        self.to_qkv = tf.keras.layers.Dense(3 * heads * head_dim, use_bias=False)
        self.unify_heads = tf.keras.layers.Dense(heads * head_dim)

    def build(self, input_shape):
        self.to_qkv.build(input_shape)
        seq_len = input_shape[1]
        unify_shape = (input_shape[0], seq_len, self.heads * self.head_dim)
        self.unify_heads.build(unify_shape)
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        qkv = self.to_qkv(x)
        qkv = tf.reshape(qkv,
                         [batch, seq_len, self.heads, 3 * self.head_dim])
        q, k, v = tf.split(qkv, 3, axis=-1)

        pad = self.window_size // 2
        k = tf.pad(k, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        v = tf.pad(v, [[0, 0], [pad, pad], [0, 0], [0, 0]])

        k_windows = tf.signal.frame(k,
                                    frame_length=self.window_size,
                                    frame_step=1,
                                    axis=1)
        v_windows = tf.signal.frame(v,
                                    frame_length=self.window_size,
                                    frame_step=1,
                                    axis=1)
        k_windows = tf.transpose(k_windows, [0, 3, 1, 2, 4])
        v_windows = tf.transpose(v_windows, [0, 3, 1, 2, 4])
        q = tf.transpose(q, [0, 2, 1, 3])

        scores = tf.einsum('bhqd,bhqkd->bhqk', q, k_windows) \
                 / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)

        attn = tf.einsum('bhqk,bhqkd->bhqd', weights, v_windows)
        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, [batch, seq_len, self.heads * self.head_dim])

        return self.unify_heads(attn)

    def get_config(self):
        config = super().get_config()
        config.update({
            "heads": self.heads,
            "head_dim": self.head_dim,
            "window_size": self.window_size
        })
        return config

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, channels, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.dropout_rate = dropout
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='silu')
        self.dense2 = tf.keras.layers.Dense(channels)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        self.dense1.build(input_shape)
        intermediate_shape = (input_shape[0], input_shape[1], self.dense1.units)
        self.dense2.build(intermediate_shape)
        self.dropout.build(intermediate_shape)
        super().build(input_shape)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dropout(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "channels": self.channels,
            "dropout": self.dropout_rate
        })
        return config

class ConvAttnBlock(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, heads, window_size, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.heads = heads
        self.window_size = window_size
        self.mlp_dim = mlp_dim
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv = DynamicConv1D(kernel_size, heads, channels)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = LocalSelfAttention(heads, channels // heads, window_size)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff = FeedForward(mlp_dim, channels)

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.conv.build(input_shape)
        self.norm2.build(input_shape)
        self.attn.build(input_shape)
        self.norm3.build(input_shape)
        self.ff.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.ff(self.norm3(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "heads": self.heads,
            "window_size": self.window_size,
            "mlp_dim": self.mlp_dim
        })
        return config


def build_model(vocab_size=32109,
                seq_len=1032,
                num_layers=27,
                channels=243,
                kernel_size=7,
                heads=3,
                window_size=53,
                mlp_dim=438):
    inputs = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, channels)(inputs)
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(seq_len, channels)(positions)
    x = x + pos_emb

    for _ in range(num_layers):
        x = ConvAttnBlock(channels, kernel_size, heads, window_size, mlp_dim)(x)

    logits = layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs=inputs, outputs=logits)

if __name__ == "__main__":
    model = build_model()
    model.summary()
