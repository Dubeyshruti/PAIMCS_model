import unittest
import tensorflow as tf
import numpy as np
from paimcs.multiscale import MultiScaleKernelFeatures

class TestMultiScaleKernelFeatures(unittest.TestCase):
    def setUp(self):
        # Set parameters for testing.
        self.input_dim = 8
        self.num_features_per_scale = 16
        self.gamma_list = [1.0, 2.0, 0.5]
        self.dropout_rate = 0.0  # Set dropout to zero for deterministic tests.
        self.batch_size = 4
        
        # Create an instance of the MultiScaleKernelFeatures layer.
        self.layer = MultiScaleKernelFeatures(
            input_dim=self.input_dim,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            dropout_rate=self.dropout_rate
        )
        
        # Create a dummy input tensor of shape (batch_size, input_dim).
        self.input_tensor = tf.random.uniform((self.batch_size, self.input_dim), dtype=tf.float16)
    
    def test_output_shape(self):
        # The expected output shape is (batch_size, num_features_per_scale * len(gamma_list)).
        output = self.layer(self.input_tensor, training=False)
        expected_features = self.num_features_per_scale * len(self.gamma_list)
        self.assertEqual(output.shape, (self.batch_size, expected_features))
    
    def test_deterministic_inference(self):
        # With dropout disabled, outputs in inference mode should be identical.
        output1 = self.layer(self.input_tensor, training=False)
        output2 = self.layer(self.input_tensor, training=False)
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)
    
    def test_dropout_training_variability(self):
        # When dropout is enabled, outputs in training mode should differ due to randomness.
        dropout_layer = MultiScaleKernelFeatures(
            input_dim=self.input_dim,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            dropout_rate=0.5
        )
        output_train1 = dropout_layer(self.input_tensor, training=True)
        output_train2 = dropout_layer(self.input_tensor, training=True)
        self.assertFalse(
            np.allclose(output_train1.numpy(), output_train2.numpy(), atol=1e-5),
            "Outputs in training mode with dropout should differ due to randomness."
        )

if __name__ == '__main__':
    unittest.main()