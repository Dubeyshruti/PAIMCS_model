import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Embedding, Dropout
from tensorflow.keras import Model
import numpy as np

# Define the Transformer model
class LM(Model):
    def __init__(self, vocab_size=32109, emb_size=243, num_heads=3, num_layers=23, ff_dim=438, dropout_rate=0.1):
        super(LM, self).__init__()
        
        self.embedding = Embedding(vocab_size, emb_size)
        self.dropout = Dropout(dropout_rate)
        
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(TransformerBlock(emb_size, num_heads, ff_dim, dropout_rate))
        
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)
        
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        
        return self.dense(x)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_heads, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=emb_size)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='silu'),
            Dense(emb_size)
        ])
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        self.dropout = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # Self-attention layer
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)  # Skip connection
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)  # Skip connection
        
        return out2
