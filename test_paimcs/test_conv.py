import unittest
import tensorflow as tf
import numpy as np
from paimcs.conv import GroupedPointwiseConv1D

class TestGroupedPointwiseConv1D(unittest.TestCase):

    def test_invalid_group_channels(self):
        # Test that a ValueError is raised when channels are not divisible by groups.
        with self.assertRaises(ValueError):
            # input_channels=5, groups=2 is invalid.
            _ = GroupedPointwiseConv1D(input_channels=5, output_channels=8, groups=2)
        with self.assertRaises(ValueError):
            # output_channels=7, groups=2 is invalid.
            _ = GroupedPointwiseConv1D(input_channels=8, output_channels=7, groups=2)

    def test_output_shape(self):
        batch_size = 4
        sequence_length = 10
        input_channels = 8
        output_channels = 16
        groups = 2
        # Use zero dropout to remove randomness.
        layer = GroupedPointwiseConv1D(input_channels=input_channels, 
                                       output_channels=output_channels, 
                                       groups=groups, 
                                       dropout_rate=0.0)
        # Create a dummy input tensor with shape (batch_size, sequence_length, input_channels).
        x = tf.random.uniform((batch_size, sequence_length, input_channels))
        output = layer(x, training=False)
        # Expected output shape: (batch_size, sequence_length, output_channels)
        self.assertEqual(output.shape, (batch_size, sequence_length, output_channels))

    def test_dropout_inference(self):
        # With dropout_rate=0.0, the output should be identical whether in training mode or not.
        batch_size = 2
        sequence_length = 5
        input_channels = 8
        output_channels = 8
        groups = 1
        layer = GroupedPointwiseConv1D(input_channels=input_channels, 
                                       output_channels=output_channels, 
                                       groups=groups, 
                                       dropout_rate=0.0)
        x = tf.random.uniform((batch_size, sequence_length, input_channels))
        output_train = layer(x, training=True)
        output_infer = layer(x, training=False)
        np.testing.assert_allclose(output_train.numpy(), output_infer.numpy(), atol=1e-5)

    def test_dropout_training_variability(self):
        # With a nonzero dropout rate, outputs in training mode should differ due to randomness.
        batch_size = 2
        sequence_length = 5
        input_channels = 8
        output_channels = 8
        groups = 1
        dropout_rate = 0.5
        layer = GroupedPointwiseConv1D(input_channels=input_channels, 
                                       output_channels=output_channels, 
                                       groups=groups, 
                                       dropout_rate=dropout_rate)
        x = tf.random.uniform((batch_size, sequence_length, input_channels))
        output1 = layer(x, training=True)
        output2 = layer(x, training=True)
        # Expect that outputs differ in training mode due to dropout.
        self.assertFalse(
            np.allclose(output1.numpy(), output2.numpy(), atol=1e-5),
            "Outputs in training mode with dropout should differ due to randomness."
        )

if __name__ == '__main__':
    unittest.main()