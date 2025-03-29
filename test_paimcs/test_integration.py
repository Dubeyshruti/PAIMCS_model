# tests/test_integration.py
import unittest
import tensorflow as tf
from paimcs.lm import paimcsLM

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Use a relatively small model configuration for integration testing.
        self.vocab_size = 100
        self.max_seq_len = 15
        self.embedding_dim = 8
        self.conv_output_channels = 8
        self.groups = 2  # Must evenly divide embedding_dim and conv_output_channels.
        self.num_features_per_scale = 4
        self.gamma_list = [1.0, 0.5]
        self.attn_dim = 16           # Total attention dimension.
        self.num_heads = 4           # Head dimension = attn_dim // num_heads.
        self.num_random_features = 4
        self.nystrom_landmarks = 4
        self.num_layers = 1
        self.dropout_rate = 0.0      # Zero dropout for deterministic integration testing.
        
        self.model = paimcsLM(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            embedding_dim=self.embedding_dim,
            conv_output_channels=self.conv_output_channels,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            attn_dim=self.attn_dim,
            num_heads=self.num_heads,
            num_random_features=self.num_random_features,
            nystrom_landmarks=self.nystrom_landmarks,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        )
        self.batch_size = 2
        self.token_seq = tf.random.uniform(
            (self.batch_size, self.max_seq_len),
            minval=0, maxval=self.vocab_size, dtype=tf.int32
        )

    def test_forward_pass(self):
        # Forward pass through the entire model.
        logits = self.model(self.token_seq, training=False)
        # For autoregressive prediction, the model outputs logits for the last token.
        self.assertEqual(logits.shape, (self.batch_size, self.vocab_size))

if __name__ == '__main__':
    unittest.main()