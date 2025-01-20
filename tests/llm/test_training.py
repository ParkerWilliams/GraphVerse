import unittest

import torch

from graphverse.data.preparation import WalkVocabulary
from graphverse.llm.training import train_model


class TestLLMTraining(unittest.TestCase):
    def test_train_model(self):
        vocab = WalkVocabulary([[1, 2, 3], [2, 3, 4]])
        training_data = torch.randint(
            0, len(vocab), (100, 10)
        )  # 100 samples, length 10

        model = train_model(
            training_data, vocab, epochs=1, batch_size=32, learning_rate=0.001
        )

        self.assertIsNotNone(model)
        self.assertEqual(model.fc_out.out_features, len(vocab))


if __name__ == "__main__":
    unittest.main()
