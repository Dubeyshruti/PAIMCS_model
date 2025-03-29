import unittest
import numpy as np
import tensorflow as tf
from paimcs.norms import RMSNorm

class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        # Initialize the layer with a sample epsilon value.
        self.layer = RMSNorm(epsilon=1e-5)

    def test_build_and_call(self):
        # Create a sample input tensor with defined last dimension.
        x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float16)
        # Call the layer to build weights and compute the normalized output.
        y = self.layer(x)
        
        # Check that the output shape matches the input shape.
        self.assertEqual(y.shape, x.shape)
        # Check that the gamma weight is created with the correct shape.
        self.assertEqual(self.layer.gamma.shape, (x.shape[-1],))
        
    def test_invalid_input_shape(self):
        # Create an input shape with an undefined last dimension.
        invalid_shape = tf.TensorShape([None])
        # The build method should raise a ValueError.
        with self.assertRaises(ValueError):
            self.layer.build(invalid_shape)

if __name__ == '__main__':
    unittest.main()