import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from .dataset import prepare_datasets
from .model_lstm import LSTMTextClassifier



# =====================
# Configurações gerais
# =====================
DATA_PATH = "projects/02_deep_learning/data/raw/SMSSpamCollection"
MODEL_OUTPUT_PATH = "projects/02_deep_learning/models/text_classifier.pt"

BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-3
MAX_LENGTH = 100

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
    return df


# =====================
# Treinamento
# =====================
def train():
    print(f"Using device: {DEVICE}")

    df = load_data()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset, val_dataset = prepare_datasets(
        df,
        tokenizer,
        max_length=MAX_LENGTH,
    )

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

    vocab_size = tokenizer.vocab_size

    model = LSTMTextClassifier(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=128,
)

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # =====================
    # Loop de treino
    # =====================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in progress_bar:

            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)


            optimizer.zero_grad()

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # =====================
        # Validação
        # =====================
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)


                outputs = model(input_ids)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        print(f"Validation accuracy: {val_accuracy:.4f}")

    # =====================
    # Salvar modelo
    # =====================
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"Model saved at {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
