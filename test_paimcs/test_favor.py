import unittest
import tensorflow as tf
import numpy as np
from paimcs.favor import FAVORProjection

class TestFAVORProjection(unittest.TestCase):
    def setUp(self):
        # Define input dimension and number of random features.
        self.input_dim = 8
        self.num_features = 16
        self.layer = FAVORProjection(input_dim=self.input_dim, num_features=self.num_features)

    def test_weight_shapes(self):
        # Build the layer with a dummy input to initialize weights.
        x = tf.random.uniform((1, self.input_dim), dtype=tf.float16)
        _ = self.layer(x)
        # Check that the weight shapes are as expected.
        self.assertEqual(self.layer.W.shape, (self.input_dim, self.num_features))
        self.assertEqual(self.layer.bias.shape, (self.num_features,))

    def test_call_output_shape(self):
        # Create a dummy input tensor with shape (batch_size, input_dim).
        batch_size = 4
        x = tf.random.uniform((batch_size, self.input_dim), dtype=tf.float16)
        output = self.layer(x)
        # tensordot(x, W) produces shape (batch_size, num_features), bias adds to that,
        # and norm_sq is broadcasted to (batch_size, 1) so the final output shape is (batch_size, num_features).
        self.assertEqual(output.shape, (batch_size, self.num_features))

    def test_call_output_values(self):
        # Check that the computed output values are positive.
        batch_size = 2
        x = tf.random.uniform((batch_size, self.input_dim), dtype=tf.float16)
        output = self.layer(x)
        # Since the layer computes exponentials, all outputs should be > 0.
        self.assertTrue(np.all(output.numpy() > 0), "Output contains non-positive values.")

if __name__ == '__main__':
    unittest.main()