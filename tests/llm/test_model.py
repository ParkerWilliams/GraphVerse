import unittest

import torch

from graphverse.llm.model import WalkTransformer


class TestLLMModel(unittest.TestCase):
    def test_walk_transformer(self):
        vocab_size = 100
        d_model = 128
        nhead = 4
        num_layers = 2
        dim_feedforward = 512

        model = WalkTransformer(vocab_size, d_model, nhead, num_layers, dim_feedforward)

        # Test forward pass
        batch_size = 32
        seq_length = 20
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model(x)

        self.assertEqual(output.shape, (batch_size, seq_length, vocab_size))


if __name__ == "__main__":
    unittest.main()
