import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification.
    """

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def prepare_datasets(
    df,
    tokenizer,
    max_length=100,
    test_size=0.2,
    random_state=42,
):
    """
    Tokenize texts and create train/validation datasets.
    """

    label_map = {"ham": 0, "spam": 1}
    labels = df["label"].map(label_map).values
    texts = df["text"].values

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    train_encodings = tokenizer(
        list(X_train),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    val_encodings = tokenizer(
        list(X_val),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    train_dataset = TextDataset(
        train_encodings,
        torch.tensor(y_train),
    )

    val_dataset = TextDataset(
        val_encodings,
        torch.tensor(y_val),
    )

    return train_dataset, val_dataset
