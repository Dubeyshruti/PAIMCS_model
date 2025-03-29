import unittest
import tensorflow as tf
import numpy as np
from paimcs.rope import rotary_embedding, apply_rotary_pos_emb

class TestRotaryEmbedding(unittest.TestCase):
    def test_rotary_embedding_shapes(self):
        head_dim = 4  # even head_dim
        seq_len = 10
        sine, cosine = rotary_embedding(head_dim, seq_len)
        # Expected shape: (1, seq_len, head_dim//2)
        self.assertEqual(sine.shape, (1, seq_len, head_dim // 2))
        self.assertEqual(cosine.shape, (1, seq_len, head_dim // 2))
    
    def test_rotary_embedding_invalid_head_dim(self):
        head_dim = 3  # odd head_dim should trigger ValueError
        seq_len = 10
        with self.assertRaises(ValueError):
            rotary_embedding(head_dim, seq_len)
            
class TestApplyRotaryPosEmb(unittest.TestCase):
    def test_output_shape(self):
        batch = 2
        num_heads = 3
        seq_len = 5
        head_dim = 4  # even head_dim
        # Create a dummy input tensor of shape (batch, num_heads, seq_len, head_dim)
        x = tf.random.uniform((batch, num_heads, seq_len, head_dim), dtype=tf.float16)
        sine, cosine = rotary_embedding(head_dim, seq_len)
        output = apply_rotary_pos_emb(x, sine, cosine)
        # The output shape should match the input shape.
        self.assertEqual(output.shape, x.shape)
    
    def test_identity_when_sin_zero_and_cos_one(self):
        batch = 1
        num_heads = 1
        seq_len = 4
        head_dim = 4  # even, so head_dim//2 == 2
        # Create a dummy input tensor with known values.
        x = tf.random.uniform((batch, num_heads, seq_len, head_dim), dtype=tf.float16)
        # Define sin tensor as zeros and cos tensor as ones.
        sin_tensor = tf.zeros((1, seq_len, head_dim // 2), dtype=tf.float16)
        cos_tensor = tf.ones((1, seq_len, head_dim // 2), dtype=tf.float16)
        output = apply_rotary_pos_emb(x, sin_tensor, cos_tensor)
        # When sin=0 and cos=1, the output should equal the input.
        self.assertTrue(
            np.allclose(x.numpy(), output.numpy(), atol=1e-2),
            "Output does not match input when sin is zero and cos is one."
        )

if __name__ == '__main__':
    unittest.main()