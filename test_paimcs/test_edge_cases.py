import unittest
import tensorflow as tf
from paimcs.embeddings import TokenRepresentation
from paimcs.conv import GroupedPointwiseConv1D

class TestEdgeCases(unittest.TestCase):
    def test_empty_token_sequence(self):
        # Test handling an empty token sequence.
        vocab_size = 50
        embedding_dim = 16
        conv_output_channels = 16
        groups = 2
        num_features_per_scale = 8
        gamma_list = [1.0, 0.5]
        dropout_rate = 0.0
        
        layer = TokenRepresentation(vocab_size, embedding_dim, conv_output_channels,
                                    groups, num_features_per_scale, gamma_list, dropout_rate)
        # Create an empty token sequence (0 batch size).
        token_seq = tf.constant([], shape=(0, 10), dtype=tf.int32)
        output = layer(token_seq, training=False)
        # Expect the batch dimension to be 0.
        self.assertEqual(output.shape[0], 0)

    def test_invalid_group_configuration(self):
        # GroupedPointwiseConv1D should raise ValueError if channels are not divisible by groups.
        with self.assertRaises(ValueError):
            _ = GroupedPointwiseConv1D(input_channels=7, output_channels=14, groups=3)

if __name__ == '__main__':
    unittest.main()