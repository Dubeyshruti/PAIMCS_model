import unittest
import numpy as np
import tensorflow as tf
from paimcs.pos_encode import PositionalEncoding

class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        # Use a modest d_model and max_len for testing.
        self.d_model = 16
        self.max_len = 50
        self.layer = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)

    def test_pos_encoding_shape_and_dtype(self):
        # The pos_encoding tensor should have shape (1, max_len, d_model)
        pos_encoding = self.layer.pos_encoding
        self.assertEqual(pos_encoding.shape, (1, self.max_len, self.d_model))
        self.assertEqual(pos_encoding.dtype, tf.float16)

    def test_call_output_shape(self):
        # Create a dummy input tensor with shape (batch_size, sequence_length, d_model)
        batch_size = 2
        seq_len = 10
        dummy_input = tf.zeros((batch_size, seq_len, self.d_model), dtype=tf.float16)
        output = self.layer(dummy_input)
        # The output shape should match the input shape.
        self.assertEqual(output.shape, dummy_input.shape)

    def test_call_output_values(self):
        # When the input is nonzero, the output should be the input plus the positional encoding.
        batch_size = 1
        seq_len = 20
        dummy_input = tf.ones((batch_size, seq_len, self.d_model), dtype=tf.float16)
        output = self.layer(dummy_input)
        # The difference between output and input should equal the slice of pos_encoding.
        expected_encoding = self.layer.pos_encoding[:, :seq_len, :]
        diff = output - dummy_input
        self.assertTrue(
            np.allclose(diff.numpy(), expected_encoding.numpy(), atol=1e-2),
            "The positional encoding was not added correctly."
        )

if __name__ == '__main__':
    unittest.main()