import unittest
import tensorflow as tf
import numpy as np
from paimcs.projection import ProjectionWithKernel

class TestProjectionWithKernel(unittest.TestCase):
    def setUp(self):
        # Define test parameters.
        self.batch_size = 2
        self.sequence_length = 5
        self.conv_channels = 8
        self.output_dim = 10
        self.groups = 2  # Must evenly divide conv_channels (8 % 2 == 0)
        self.num_features_per_scale = 4
        self.gamma_list = [1.0, 0.5]
        self.dropout_rate = 0.0  # Use zero dropout for deterministic inference
        
        # Create an instance of ProjectionWithKernel.
        self.layer = ProjectionWithKernel(
            output_dim=self.output_dim,
            conv_channels=self.conv_channels,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            dropout_rate=self.dropout_rate
        )
        
        # Create a dummy input tensor with shape (batch_size, sequence_length, conv_channels).
        self.input_tensor = tf.random.uniform(
            (self.batch_size, self.sequence_length, self.conv_channels),
            dtype=tf.float32
        )

    def test_output_shape(self):
        # Run the layer in inference mode.
        output = self.layer(self.input_tensor, training=False)
        # Expected output shape is (batch_size, output_dim).
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.output_dim))

    def test_inference_determinism(self):
        # With dropout disabled, outputs should be the same across calls in inference mode.
        output1 = self.layer(self.input_tensor, training=False)
        output2 = self.layer(self.input_tensor, training=False)
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)

    def test_dropout_effect_in_training(self):
        # When dropout is enabled, training mode outputs should vary.
        layer_dropout = ProjectionWithKernel(
            output_dim=self.output_dim,
            conv_channels=self.conv_channels,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            dropout_rate=0.5  # Nonzero dropout for training variability
        )
        out_train1 = layer_dropout(self.input_tensor, training=True)
        out_train2 = layer_dropout(self.input_tensor, training=True)
        self.assertFalse(
            np.allclose(out_train1.numpy(), out_train2.numpy(), atol=1e-5),
            "Training outputs with dropout should differ due to randomness."
        )

if __name__ == '__main__':
    unittest.main()