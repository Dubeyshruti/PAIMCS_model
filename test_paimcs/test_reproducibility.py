import unittest
import tensorflow as tf
import numpy as np
from paimcs.lm import paimcsLM

class TestReproducibility(unittest.TestCase):
    def test_reproducible_output(self):
        # Set a fixed random seed and instantiate the model.
        tf.random.set_seed(42)
        vocab_size = 50
        max_seq_len = 20
        embedding_dim = 16
        conv_output_channels = 16
        groups = 2
        num_features_per_scale = 8
        gamma_list = [1.0, 0.5]
        attn_dim = 16
        num_heads = 4
        num_random_features = 8
        nystrom_landmarks = 8
        num_layers = 1
        dropout_rate = 0.0
        
        model1 = paimcsLM(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            conv_output_channels=conv_output_channels,
            groups=groups,
            num_features_per_scale=num_features_per_scale,
            gamma_list=gamma_list,
            attn_dim=attn_dim,
            num_heads=num_heads,
            num_random_features=num_random_features,
            nystrom_landmarks=nystrom_landmarks,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        token_seq1 = tf.random.uniform(
            (2, max_seq_len), minval=0, maxval=vocab_size, dtype=tf.int32
        )
        output1 = model1(token_seq1, training=False)
        
        # Reset the seed and create a new model instance.
        tf.random.set_seed(42)
        model2 = paimcsLM(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            conv_output_channels=conv_output_channels,
            groups=groups,
            num_features_per_scale=num_features_per_scale,
            gamma_list=gamma_list,
            attn_dim=attn_dim,
            num_heads=num_heads,
            num_random_features=num_random_features,
            nystrom_landmarks=nystrom_landmarks,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        token_seq2 = tf.random.uniform(
            (2, max_seq_len), minval=0, maxval=vocab_size, dtype=tf.int32
        )
        output2 = model2(token_seq2, training=False)
        
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)

if __name__ == '__main__':
    unittest.main()