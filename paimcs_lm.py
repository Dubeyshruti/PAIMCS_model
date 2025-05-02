import tensorflow as tf
from tensorflow.keras import layers, Model

class DynamicConv1D(layers.Layer):
    def __init__(self, kernel_size, num_heads, channels, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        # Predict per-position, per-head kernels
        self.kernel_predict = layers.Dense(kernel_size * num_heads)
        self.padding = kernel_size // 2

    def build(self, input_shape):
        # no additional weights
        super().build(input_shape)

    def call(self, x):  # x: [batch, seq_len, channels]
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        # Predict kernels: [batch, seq_len, num_heads, kernel_size]
        kernels = self.kernel_predict(x)
        kernels = tf.reshape(kernels, [batch, seq_len, self.num_heads, self.kernel_size])
        x_heads = tf.reshape(x, [batch, seq_len, self.num_heads, self.head_dim])
        x_padded = tf.pad(x_heads, [[0,0],[self.padding,self.padding],[0,0],[0,0]])
        # Convolution via einsum over sliding window
        outputs = []
        for i in range(self.kernel_size):
            slice_i = x_padded[:, i:i+seq_len, :, :]
            weight_i = tf.expand_dims(kernels[:, :, :, i], -1)
            outputs.append(weight_i * slice_i)
        out = tf.add_n(outputs)
        out = tf.reshape(out, [batch, seq_len, self.channels])
        return out

class LocalSelfAttention(layers.Layer):
    def __init__(self, heads, head_dim, window_size, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.to_qkv = layers.Dense(3 * heads * head_dim, use_bias=False)
        self.unify_heads = layers.Dense(heads * head_dim)

    def build(self, input_shape):
        # window_size must be odd
        super().build(input_shape)

    def call(self, x):  # x: [batch, seq_len, channels]
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        qkv = self.to_qkv(x)
        qkv = tf.reshape(qkv, [batch, seq_len, self.heads, 3 * self.head_dim])
        q, k, v = tf.split(qkv, 3, axis=-1)
        pad = self.window_size // 2
        # pad along sequence axis
        k = tf.pad(k, [[0,0],[pad,pad],[0,0],[0,0]])
        v = tf.pad(v, [[0,0],[pad,pad],[0,0],[0,0]])
        # Collect windows via tf.signal.frame for efficiency
        k_windows = tf.signal.frame(k, frame_length=self.window_size, frame_step=1, axis=1)
        v_windows = tf.signal.frame(v, frame_length=self.window_size, frame_step=1, axis=1)
        # k_windows: [batch, seq_len, window_size, heads, head_dim]
        # Move heads before window for einsum
        k_windows = tf.transpose(k_windows, [0,3,1,2,4])  # [batch, heads, seq_len, window_size, head_dim]
        v_windows = tf.transpose(v_windows, [0,3,1,2,4])
        q = tf.transpose(q, [0,2,1,3])  # [batch, heads, seq_len, head_dim]
        # Compute attention scores
        scores = tf.einsum('bhqd,bhqkd->bhqk', q, k_windows) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        attn = tf.einsum('bhqk,bhqkd->bhqd', weights, v_windows)
        attn = tf.transpose(attn, [0,2,1,3])  # [batch, seq_len, heads, head_dim]
        attn = tf.reshape(attn, [batch, seq_len, self.heads * self.head_dim])
        return self.unify_heads(attn)

class FeedForward(layers.Layer):
    def __init__(self, hidden_dim, channels, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(channels)
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dropout(x)

class ConvAttnBlock(layers.Layer):
    def __init__(self, channels, kernel_size, heads, window_size, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.conv = DynamicConv1D(kernel_size, heads, channels)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = LocalSelfAttention(heads, channels//heads, window_size)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.ff = FeedForward(mlp_dim, channels)

    def call(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.ff(self.norm3(x))
        return x


def build_model(vocab_size,
                seq_len=1032,
                num_layers=27,
                channels=243,
                kernel_size=7,
                heads=3,
                window_size=53,
                mlp_dim=438):
    inputs = layers.Input(shape=(seq_len,), dtype=tf.int32)
    x = layers.Embedding(vocab_size, channels)(inputs)
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(seq_len, channels)(positions)
    x = x + pos_emb
    for _ in range(num_layers):
        x = ConvAttnBlock(channels, kernel_size, heads, window_size, mlp_dim)(x)
    logits = layers.Dense(vocab_size)(x)
    return Model(inputs=inputs, outputs=logits)
