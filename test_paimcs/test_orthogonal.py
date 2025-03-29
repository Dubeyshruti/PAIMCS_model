import unittest
import tensorflow as tf
import numpy as np
import math
from paimcs.orthogonal import OrthogonalRandomFeaturesTF

class TestOrthogonalRandomFeaturesTF(unittest.TestCase):
    def setUp(self):
        # Define common parameters for testing.
        self.input_dim = 8
        self.num_features = 16
        self.gamma = 1.0
        self.dropout_rate = 0.0  # Use no dropout for most tests.
        self.batch_size = 4
        self.layer = OrthogonalRandomFeaturesTF(
            input_dim=self.input_dim,
            num_features=self.num_features,
            gamma=self.gamma,
            dropout_rate=self.dropout_rate
        )
        # Create a dummy input tensor.
        self.input_tensor = tf.random.uniform(
            (self.batch_size, self.input_dim), dtype=tf.float16
        )
    
    def test_weight_and_bias_shapes(self):
        # Forward pass builds the layer.
        _ = self.layer(self.input_tensor, training=False)
        # Check that W has shape (input_dim, num_features)
        self.assertEqual(self.layer.W.shape, (self.input_dim, self.num_features))
        # Check that b has shape (num_features,)
        self.assertEqual(self.layer.b.shape, (self.num_features,))
    
    def test_output_shape(self):
        # Forward pass with dropout disabled.
        output = self.layer(self.input_tensor, training=False)
        # Expected output shape is (batch_size, num_features)
        self.assertEqual(output.shape, (self.batch_size, self.num_features))
    
    def test_output_value_range(self):
        # The layer computes:
        #    rff = sqrt(2/num_features) * cos(projection)
        # so the absolute value should be at most sqrt(2/num_features)
        rff_scale = math.sqrt(2.0 / self.num_features)
        output = self.layer(self.input_tensor, training=False)
        output_np = output.numpy()
        # Allow a small numerical tolerance.
        self.assertTrue(
            np.all(np.abs(output_np) <= rff_scale + 1e-2),
            "Output values exceed the expected range based on scaling."
        )
    
    def test_dropout_behavior(self):
        # Test that dropout is applied in training mode.
        dropout_rate = 0.5
        layer_dropout = OrthogonalRandomFeaturesTF(
            input_dim=self.input_dim,
            num_features=self.num_features,
            gamma=self.gamma,
            dropout_rate=dropout_rate
        )
        # Build the layer.
        _ = layer_dropout(self.input_tensor, training=False)
        # Get outputs in training and inference mode.
        output_infer = layer_dropout(self.input_tensor, training=False)
        output_train = layer_dropout(self.input_tensor, training=True)
        # With dropout enabled, the outputs should differ.
        self.assertFalse(
            np.allclose(output_infer.numpy(), output_train.numpy(), atol=1e-3),
            "Dropout was not applied during training mode as expected."
        )

if __name__ == '__main__':
    unittest.main()