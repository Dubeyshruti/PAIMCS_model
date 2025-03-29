import unittest
import tensorflow as tf
import time
from paimcs.lm import paimcsLM

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50
        self.max_seq_len = 20
        self.embedding_dim = 16
        self.conv_output_channels = 16
        self.groups = 2
        self.num_features_per_scale = 8
        self.gamma_list = [1.0, 0.5]
        self.attn_dim = 16
        self.num_heads = 4
        self.num_random_features = 8
        self.nystrom_landmarks = 8
        self.num_layers = 1
        self.dropout_rate = 0.0
        
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
        self.batch_size = 4
        self.token_seq = tf.random.uniform(
            (self.batch_size, self.max_seq_len),
            minval=0, maxval=self.vocab_size, dtype=tf.int32
        )

    def test_forward_pass_time(self):
        start_time = time.time()
        _ = self.model(self.token_seq, training=False)
        elapsed = time.time() - start_time
        # Set an arbitrary threshold (e.g., 0.5 seconds).
        self.assertLess(elapsed, 0.5, f"Forward pass took too long: {elapsed:.3f} seconds.")

if __name__ == '__main__':
    unittest.main()