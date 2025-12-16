import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from .dataset import prepare_datasets
from .model_lstm import LSTMTextClassifier



# =====================
# Configurações
# =====================
DATA_PATH = "projects/02_deep_learning/data/raw/SMSSpamCollection"
MODEL_PATH = "projects/02_deep_learning/models/text_classifier.pt"

BATCH_SIZE = 32
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
# Avaliação
# =====================
def evaluate():
    print(f"Using device: {DEVICE}")

    df = load_data()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    _, val_dataset = prepare_datasets(
        df,
        tokenizer,
        max_length=MAX_LENGTH,
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

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)


            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Ham", "Spam"]))

    # =====================
    # Confusion Matrix
    # =====================
    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Ham", "Spam"],
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — SMS Spam Classification")
    plt.show()


if __name__ == "__main__":
    evaluate()
