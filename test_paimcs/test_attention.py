import unittest
import tensorflow as tf
import numpy as np
from paimcs.attention import MultiHeadFAVORAttention

class TestMultiHeadFAVORAttention(unittest.TestCase):
    def setUp(self):
        # Define parameters for testing.
        self.num_heads = 4
        self.attn_dim = 16        # Total attention dimension; head_dim = attn_dim // num_heads = 4.
        self.num_random_features = 8
        self.conv_channels = 16     # Should be divisible by groups.
        self.groups = 4             # 16 channels / 4 groups = 4 channels per group.
        self.num_features_per_scale = 8
        self.gamma_list = [1.0, 0.5]
        self.nystrom_landmarks = 8
        self.dropout_rate = 0.0     # Zero dropout for deterministic inference.
        self.batch_size = 2
        self.seq_len = 10
        
        # Create an instance of MultiHeadFAVORAttention.
        self.layer = MultiHeadFAVORAttention(
            num_heads=self.num_heads,
            attn_dim=self.attn_dim,
            num_random_features=self.num_random_features,
            conv_channels=self.conv_channels,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            nystrom_landmarks=self.nystrom_landmarks,
            dropout_rate=self.dropout_rate
        )
        
        # Create a dummy input tensor with shape (batch_size, seq_len, attn_dim).
        self.input_tensor = tf.random.uniform(
            (self.batch_size, self.seq_len, self.attn_dim), dtype=tf.float16
        )
    
    def test_output_shape(self):
        output = self.layer(self.input_tensor, training=False)
        # Expected output shape: (batch_size, seq_len, attn_dim)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.attn_dim))
    
    def test_inference_determinism(self):
        # With dropout disabled, repeated inference calls should yield identical outputs.
        output1 = self.layer(self.input_tensor, training=False)
        output2 = self.layer(self.input_tensor, training=False)
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)
    
    def test_no_nan_values(self):
        output = self.layer(self.input_tensor, training=False)
        self.assertFalse(np.isnan(output.numpy()).any(), "Output contains NaN values.")
    
    def test_dropout_variability(self):
        # Create a new instance with a nonzero dropout rate.
        layer_dropout = MultiHeadFAVORAttention(
            num_heads=self.num_heads,
            attn_dim=self.attn_dim,
            num_random_features=self.num_random_features,
            conv_channels=self.conv_channels,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            nystrom_landmarks=self.nystrom_landmarks,
            dropout_rate=0.5  # Nonzero dropout to test stochasticity.
        )
        output_train1 = layer_dropout(self.input_tensor, training=True)
        output_train2 = layer_dropout(self.input_tensor, training=True)
        self.assertFalse(
            np.allclose(output_train1.numpy(), output_train2.numpy(), atol=1e-5),
            "Training outputs with dropout should differ due to randomness."
        )

if __name__ == '__main__':
    unittest.main()