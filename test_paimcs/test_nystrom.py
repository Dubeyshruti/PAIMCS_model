import unittest
import tensorflow as tf
import numpy as np
from paimcs.nystrom import NystromFeatures

class TestNystromFeatures(unittest.TestCase):
    def setUp(self):
        # Define parameters for testing.
        self.num_landmarks = 5
        self.gamma = 1.0
        self.input_dim = 10
        self.batch_size = 4
        
        # Create an instance of the layer.
        self.layer = NystromFeatures(num_landmarks=self.num_landmarks, gamma=self.gamma)
        # Create a dummy input tensor.
        self.input_tensor = tf.random.uniform((self.batch_size, self.input_dim), dtype=tf.float16)
    
    def test_landmarks_shape(self):
        # A forward pass will trigger the build method.
        _ = self.layer(self.input_tensor)
        # Check that the landmarks weight is created with the correct shape.
        self.assertEqual(self.layer.landmarks.shape, (self.num_landmarks, self.input_dim))
    
    def test_call_output_shape(self):
        output = self.layer(self.input_tensor)
        # Expected output shape is (batch_size, num_landmarks)
        self.assertEqual(output.shape, (self.batch_size, self.num_landmarks))
    
    def test_output_value_range(self):
        output = self.layer(self.input_tensor)
        output_np = output.numpy()
        # Since features are computed using exp(-gamma * distance^2),
        # all output values should be positive and at most 1.
        self.assertTrue(np.all(output_np > 0), "Output values must be positive.")
        self.assertTrue(np.all(output_np <= 1.0), "Output values must be <= 1.0.")
        
if __name__ == '__main__':
    unittest.main()