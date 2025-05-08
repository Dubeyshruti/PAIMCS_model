import tensorflow as tf

class LocalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, heads, head_dim, window_size, **kwargs):
        super().__init__(**kwargs)
        assert window_size % 2 == 1, "window_size must be odd"
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.to_qkv = tf.keras.layers.Dense(3 * self.heads * self.head_dim, use_bias=False)
        self.unify_heads = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        B, L, _ = tf.unstack(tf.shape(x))
        pad = self.window_size // 2

        qkv = self.to_qkv(x)  # [B, L, 3*H*D]
        qkv = tf.reshape(qkv, [B, L, self.heads, 3 * self.head_dim])
        q, k, v = tf.split(qkv, 3, axis=-1)  # Each: [B, L, H, D]
        q = tf.transpose(q, [0, 2, 1, 3])    # [B, H, L, D]

        # Pad and unfold key/value
        k = tf.pad(k, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        v = tf.pad(v, [[0, 0], [pad, pad], [0, 0], [0, 0]])

        k_windows = []
        v_windows = []
        for i in range(self.window_size):
            k_slice = k[:, i:i + L, :, :]  # [B, L, H, D]
            v_slice = v[:, i:i + L, :, :]
            k_windows.append(k_slice)
            v_windows.append(v_slice)

        # Stack into [B, H, L, W, D]
        k_stack = tf.stack(k_windows, axis=1)  # [B, W, L, H, D]
        v_stack = tf.stack(v_windows, axis=1)  # [B, W, L, H, D]
        k_stack = tf.transpose(k_stack, [0, 3, 2, 1, 4])  # -> [B, H, L, W, D]
        v_stack = tf.transpose(v_stack, [0, 3, 2, 1, 4])

        # Attention: [B, H, L, D] x [B, H, L, W, D] -> [B, H, L, W]
        score = tf.einsum('bhld,bhlwd->bhlw', q, k_stack)
        score /= tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(score, axis=-1)  # [B, H, L, W]

        # Apply weights: [B, H, L, W] x [B, H, L, W, D] -> [B, H, L, D]
        out = tf.einsum('bhlw,bhlwd->bhld', weights, v_stack)
        out = tf.transpose(out, [0, 2, 1, 3])  # [B, L, H, D]
        out = tf.reshape(out, [B, L, self.heads * self.head_dim])  # [B, L, H*D]
        return self.unify_heads(out)

class ConvAttnBlock(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, heads, window_size, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.heads = heads
        self.window_size = window_size
        self.mlp_dim = mlp_dim
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.conv = tf.keras.layers.Conv1D(channels, kernel_size, padding='same')
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.attn = LocalSelfAttention(heads, head_dim=channels // heads, window_size=window_size)
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation='gelu'),
            tf.keras.layers.Dense(channels)
        ])

    def build(self, input_shape):
        self.conv.build(input_shape)
        self.attn.build(input_shape)
        self.mlp.build(input_shape)

    def call(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
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
    pos_emb = tf.keras.layers.Embedding(seq_len, channels)(positions)
    x = x + pos_emb

    for _ in range(num_layers):
        x = ConvAttnBlock(channels, kernel_size, heads, window_size, mlp_dim)(x)

    logits = tf.keras.layers.Dense(vocab_size, use_bias=False)(x)
    return tf.keras.Model(inputs=inputs, outputs=logits)

if __name__ == "__main__":
    model = build_model()
    model.build(input_shape=(None, None))  # dynamic batch & seq_len
    model.summary()
