import unittest
import tensorflow as tf
import numpy as np
from paimcs.lm import paimcsLM

class TestPaimcsLM(unittest.TestCase):
    def setUp(self):
        # Define model parameters.
        self.vocab_size = 50
        self.max_seq_len = 20
        self.embedding_dim = 16
        self.conv_output_channels = 16
        self.groups = 2  # Must evenly divide embedding_dim and conv_output_channels.
        self.num_features_per_scale = 8
        self.gamma_list = [1.0, 0.5]  # Two scales → final channel dim = conv_output_channels * 2.
        self.attn_dim = 16           # Total attention dimension.
        self.num_heads = 4           # Head dimension = attn_dim // num_heads = 4.
        self.num_random_features = 8
        self.nystrom_landmarks = 8
        self.num_layers = 2
        self.dropout_rate = 0.0      # Zero dropout for deterministic tests.
        
        # Instantiate the model.
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
        
        # Create a dummy token sequence with shape (batch_size, seq_len)
        self.batch_size = 4
        self.seq_len = self.max_seq_len
        self.token_seq = tf.random.uniform(
            (self.batch_size, self.seq_len), minval=0, maxval=self.vocab_size, dtype=tf.int32
        )
    
    def test_output_shape(self):
        # The model produces logits based on the last token, so output shape is (batch_size, vocab_size)
        logits = self.model(self.token_seq, training=False)
        self.assertEqual(logits.shape, (self.batch_size, self.vocab_size))
    
    def test_inference_determinism(self):
        # With dropout disabled, repeated inference calls should yield identical outputs.
        output1 = self.model(self.token_seq, training=False)
        output2 = self.model(self.token_seq, training=False)
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)
    
    def test_training_dropout_variability(self):
        # When dropout is enabled, outputs in training mode should differ due to randomness.
        model_dropout = paimcsLM(
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
            dropout_rate=0.5  # Nonzero dropout for training variability.
        )
        out_train1 = model_dropout(self.token_seq, training=True)
        out_train2 = model_dropout(self.token_seq, training=True)
        self.assertFalse(
            np.allclose(out_train1.numpy(), out_train2.numpy(), atol=1e-5),
            "Training outputs with dropout should differ due to randomness."
        )

if __name__ == '__main__':
    unittest.main()