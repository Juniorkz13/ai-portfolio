import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """
    Simple neural network for text classification using embeddings.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=2,
        dropout=0.3,
    ):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embeddings = self.embedding(input_ids)
        pooled = embeddings.mean(dim=1)  # Global Average Pooling
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
