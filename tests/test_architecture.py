"""
Pruebas unitarias para el dominio arquitectónico.
"""
import unittest
import torch
from domain.architecture import (
    precompute_freqs_cis,
    MultiHeadAttentionRoPE,
    BERT
)


class TestAttentionArchitecture(unittest.TestCase):
    """Test Attention Layer."""
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 64
        self.num_heads = 4
        self.d_k = self.hidden_size // self.num_heads
        self.vocab_size = 1000

    def test_precompute_freqs_cis_shapes(self):
        """Test Pre-Compute Freqs."""
        cos, sin = precompute_freqs_cis(self.d_k, self.seq_len)
        expected_shape = (1, 1, self.seq_len, self.d_k)
        self.assertEqual(cos.shape, expected_shape)
        self.assertEqual(sin.shape, expected_shape)

    def test_multihead_attention_rope_output_shape(self):
        """Test Multi Head Attention."""
        attention = MultiHeadAttentionRoPE(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            max_len=self.seq_len
            )
        q = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        k = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        v = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_len)

        output = attention(q, k, v, mask)
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.hidden_size)
            )

    def test_bert_encoder_flow(self):
        """Test Bert Encoder."""
        model = BERT(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            num_heads=self.num_heads,
            max_len=self.seq_len)
        model.eval()
        input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        segment_ids = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)
        output = model(input_ids, segment_ids)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))


if __name__ == '__main__':
    unittest.main()
