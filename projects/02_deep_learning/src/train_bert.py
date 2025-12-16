import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =====================
# Configurações
# =====================
DATA_PATH = "projects/02_deep_learning/data/raw/SMSSpamCollection"
MODEL_NAME = "bert-base-uncased"
MODEL_OUTPUT_PATH = "projects/02_deep_learning/models/bert_spam_classifier.pt"

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Carregar dados
# =====================
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
    )
    label_map = {"ham": 0, "spam": 1}
    df["label"] = df["label"].map(label_map)
    return df


# =====================
# Dataset simples
# =====================
class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# =====================
# Treinamento
# =====================
def train():
    print(f"Using device: {DEVICE}")

    df = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        list(X_train),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    val_encodings = tokenizer(
        list(X_val),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_dataset = BertDataset(train_encodings, y_train.values)
    val_dataset = BertDataset(val_encodings, y_val.values)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # =====================
    # Loop de treino
    # =====================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in progress:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_loss:.4f}")

    # =====================
    # Salvar modelo
    # =====================
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"Model saved at {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
