import unittest
import numpy as np
import tensorflow as tf
from paimcsLM import (
    RMSNorm, PositionalEncoding, rotary_embedding, apply_rotary_pos_emb, FAVORProjection,
    OrthogonalRandomFeaturesTF, MultiScaleKernelFeatures, GroupedPointwiseConv1D, TokenRepresentation,
    MultiHeadFAVORAttention, KernelLMBlock, paimcsLm
)

class TestPaimcsLmModel(unittest.TestCase):
    def setUp(self):
        # Define hyperparameters for testing (adjust as needed)
        self.vocab_size = 31542
        self.max_seq_len = 678
        self.embedding_dim = 360
        self.token_input_dim = 339
        self.groups = 3
        self.num_features_per_scale = 120          # Used for TokenRepresentation output.
        self.gamma_list = [99e-5, 0.099, 9.9]
        self.num_heads = 3
        self.num_random_features = 87
        self.num_layers = 3
        self.dropout_rate = 0.099

        # Instantiate the model from paimcsLM.py
        # Here, we pass self.num_features_per_scale for the token representation.
        self.model = paimcsLm(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            embedding_dim=self.embedding_dim,
            token_input_dim=self.token_input_dim,
            groups=self.groups,
            num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list,
            num_heads=self.num_heads,
            num_random_features=self.num_random_features,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        )
        # Create dummy token input: shape (batch_size, seq_len)
        self.batch_size = 4
        self.dummy_tokens = tf.constant(
            np.random.randint(0, self.vocab_size, size=(self.batch_size, self.max_seq_len)),
            dtype=tf.int32
        )

    def test_rmsnorm(self):
        layer = RMSNorm()
        x = tf.random.uniform((4, 10, 64), dtype=tf.float16)
        y = layer(x)
        self.assertEqual(x.shape, y.shape)

    def test_positional_encoding(self):
        hidden_dim = 64
        layer = PositionalEncoding(hidden_dim, max_len=self.max_seq_len)
        x = tf.random.uniform((4, self.max_seq_len, hidden_dim), dtype=tf.float16)
        y = layer(x)
        self.assertEqual(x.shape, y.shape)

    def test_rotary_embedding(self):
        head_dim = 32
        sin, cos = rotary_embedding(head_dim, self.max_seq_len)
        self.assertEqual(sin.shape, (1, self.max_seq_len, head_dim // 2))
        self.assertEqual(cos.shape, (1, self.max_seq_len, head_dim // 2))

    def test_apply_rotary_pos_emb(self):
        # Create a dummy tensor for one head.
        x = tf.random.uniform((2, 4, self.max_seq_len, 32), dtype=tf.float16)
        sin, cos = rotary_embedding(32, self.max_seq_len)
        y = apply_rotary_pos_emb(x, sin, cos)
        self.assertEqual(x.shape, y.shape)

    def test_favor_projection(self):
        layer = FAVORProjection(input_dim=32, num_features=16)
        x = tf.random.uniform((2, 4, 10, 32), dtype=tf.float16)
        y = layer(x)
        self.assertEqual(y.shape, (2, 4, 10, 16))

    def test_orthogonal_random_features(self):
        layer = OrthogonalRandomFeaturesTF(input_dim=32, num_features=16, gamma=1.0, dropout_rate=0.0)
        x = tf.random.uniform((2, 10, 32), dtype=tf.float16)
        y = layer(x, training=False)
        self.assertEqual(y.shape, (2, 10, 16))

    def test_multiscale_kernel_features(self):
        layer = MultiScaleKernelFeatures(input_dim=32, num_features_per_scale=8, gamma_list=self.gamma_list, dropout_rate=0.0)
        x = tf.random.uniform((2, 10, 32), dtype=tf.float16)
        y = layer(x, training=False)
        self.assertEqual(y.shape, (2, 10, 8 * len(self.gamma_list)))

    def test_grouped_conv1d(self):
        layer = GroupedPointwiseConv1D(input_channels=32, output_channels=64, groups=1, dropout_rate=0.099)
        x = tf.random.uniform((2, 10, 32), dtype=tf.float16)
        y = layer(x, training=False)
        self.assertEqual(y.shape, (2, 10, 64))

    def test_token_representation(self):
        layer = TokenRepresentation(
            vocab_size=self.vocab_size,
            token_input_dim=self.token_input_dim, groups=self.groups,
            max_seq_len=self.max_seq_len, num_features_per_scale=self.num_features_per_scale,
            gamma_list=self.gamma_list, dropout_rate=self.dropout_rate
        )
        y = layer(self.dummy_tokens, training=False)
        expected_dim = self.num_features_per_scale * len(self.gamma_list)
        self.assertEqual(y.shape, (self.batch_size, self.max_seq_len, expected_dim))

    def test_multihead_favor_attention(self):
        layer = MultiHeadFAVORAttention(
            num_heads=self.num_heads, attn_dim=self.embedding_dim,
            num_random_features=self.num_random_features, groups=self.groups,
            gamma_list=self.gamma_list, dropout_rate=self.dropout_rate, proj_type="conv"
        )
        x = tf.random.uniform((2, self.max_seq_len, self.embedding_dim), dtype=tf.float16)
        y = layer(x, training=False)
        self.assertEqual(y.shape, (2, self.max_seq_len, self.embedding_dim))

    def test_kernel_lm_block(self):
        # Note: KernelLMBlock expects 'num_features_per_scale' parameter; we pass the attention projection variant here.
        block = KernelLMBlock(
            attn_dim=self.embedding_dim, num_heads=self.num_heads, num_random_features=self.num_random_features,
            groups=self.groups, num_features_per_scale=self.num_features_per_scale, gamma_list=self.gamma_list,
            dropout_rate=self.dropout_rate, proj_type="conv"
        )
        x = tf.random.uniform((2, self.max_seq_len, self.embedding_dim), dtype=tf.float16)
        y = block(x, training=False)
        self.assertEqual(y.shape, (2, self.max_seq_len, self.embedding_dim))

    def test_model_output_shape(self):
        logits = self.model(self.dummy_tokens, training=False)
        expected_shape = (self.batch_size, self.vocab_size)
        self.assertEqual(logits.shape, expected_shape, f"Expected shape {expected_shape}, got {logits.shape}")

    def test_model_dtypes(self):
        logits = self.model(self.dummy_tokens, training=False)
        self.assertEqual(logits.dtype, tf.float16, f"Expected dtype tf.float16, got {logits.dtype}")

    def test_model_train_vs_inference(self):
        _ = self.model(self.dummy_tokens, training=True)
        _ = self.model(self.dummy_tokens, training=False)

    def test_model_consistency(self):
        logits1 = self.model(self.dummy_tokens, training=False)
        logits2 = self.model(self.dummy_tokens, training=False)
        self.assertEqual(logits1.shape, logits2.shape, "Output shapes differ between inference calls")

    def test_tflite_conversion(self):
        @tf.function(input_signature=[tf.TensorSpec([None, self.max_seq_len], tf.int32)])
        def inference_fn(token_seq):
            return self.model(token_seq, training=False)
        concrete_func = inference_fn.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops.add(tf.lite.OpsSet.SELECT_TF_OPS)
        try:
            tflite_model = converter.convert()
        except Exception as e:
            self.fail(f"TFLite conversion failed: {e}")
        self.assertTrue(len(tflite_model) > 0, "TFLite conversion resulted in empty model data.")

if __name__ == "__main__":
    unittest.main()
