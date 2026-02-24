import torch
import torch.nn as nn


class LSTMTextClassifier(nn.Module):
    """
    LSTM-based text classifier for NLP tasks.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embeddings = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embeddings)
        x = hidden[-1]  # last layer hidden state
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
