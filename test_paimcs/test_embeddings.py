import unittest
import tensorflow as tf
import numpy as np
from paimcs.embeddings import TokenRepresentation

class TestTokenRepresentation(unittest.TestCase):
    def setUp(self):
        # Define parameters for testing.
        self.vocab_size = 1000
        self.embedding_dim = 16
        self.conv_output_channels = 16
        self.groups = 2  # Both embedding_dim and conv_output_channels must be divisible by groups.
        self.num_features_per_scale = 8
        self.gamma_list = [1.0, 0.5]  # Two scales.
        self.dropout_rate = 0.0  # Use zero dropout for deterministic inference.
        self.batch_size = 4
        self.seq_len = 10

        # Create an instance of the TokenRepresentation layer.
        self.layer = TokenRepresentation(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            conv_output_channels=self.conv_output_channels,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            dropout_rate=self.dropout_rate
        )
        
        # Create a dummy token sequence with integer token IDs.
        self.token_seq = tf.random.uniform(
            (self.batch_size, self.seq_len), minval=0, maxval=self.vocab_size, dtype=tf.int32
        )

    def test_output_shape(self):
        # Run the layer in inference mode.
        output = self.layer(self.token_seq, training=False)
        # Expected final feature dimension equals num_features_per_scale * number of scales.
        expected_feature_dim = self.num_features_per_scale * len(self.gamma_list)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, expected_feature_dim))

    def test_inference_determinism(self):
        # With dropout disabled, repeated calls in inference mode should produce identical outputs.
        output1 = self.layer(self.token_seq, training=False)
        output2 = self.layer(self.token_seq, training=False)
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)

if __name__ == '__main__':
    unittest.main()