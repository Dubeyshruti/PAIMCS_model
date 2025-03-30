import unittest
import tensorflow as tf
import numpy as np
from paimcs.performer import KernelLMBlock

class TestKernelLMBlock(unittest.TestCase):
    def setUp(self):
        # Define parameters for the KernelLMBlock.
        self.attn_dim = 16
        self.num_heads = 4
        self.num_random_features = 8
        self.conv_channels = 16         # Should match the attn_dim for this test.
        self.groups = 4                 # Must evenly divide conv_channels.
        self.num_features_per_scale = 8
        self.gamma_list = [1.0, 0.5]
        self.nystrom_landmarks = 8
        self.dropout_rate = 0.0         # Zero dropout for deterministic tests.
        
        # Create an instance of KernelLMBlock with zero dropout.
        self.block = KernelLMBlock(
            attn_dim=self.attn_dim,
            num_heads=self.num_heads,
            num_random_features=self.num_random_features,
            conv_channels=self.conv_channels,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            nystrom_landmarks=self.nystrom_landmarks,
            dropout_rate=self.dropout_rate
        )
        
        # Create a dummy input tensor with shape (batch_size, seq_len, attn_dim).
        self.batch_size = 2
        self.seq_len = 10
        self.input_tensor = tf.random.uniform(
            (self.batch_size, self.seq_len, self.attn_dim), dtype=tf.float16
        )
    
    def test_output_shape(self):
        # Verify that the output shape matches the input shape.
        output = self.block(self.input_tensor, training=False)
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_inference_determinism(self):
        # With dropout disabled, repeated inference calls should yield identical outputs.
        output1 = self.block(self.input_tensor, training=False)
        output2 = self.block(self.input_tensor, training=False)
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)

    def test_dropout_variability(self):
        # Create a new instance with a nonzero dropout rate to test stochastic behavior.
        block_dropout = KernelLMBlock(
            attn_dim=self.attn_dim,
            num_heads=self.num_heads,
            num_random_features=self.num_random_features,
            conv_channels=self.conv_channels,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            nystrom_landmarks=self.nystrom_landmarks,
            dropout_rate=0.5
        )
        output_train1 = block_dropout(self.input_tensor, training=True)
        output_train2 = block_dropout(self.input_tensor, training=True)
        self.assertFalse(
            np.allclose(output_train1.numpy(), output_train2.numpy(), atol=1e-5),
            "Outputs in training mode with dropout should differ due to randomness."
        )

if __name__ == '__main__':
    unittest.main()