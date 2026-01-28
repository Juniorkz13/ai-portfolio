import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# =====================
# Configurações
# =====================
DATA_PATH = "projects/02_deep_learning/data/raw/SMSSpamCollection"
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "projects/02_deep_learning/models/bert_spam_classifier.pt"

BATCH_SIZE = 16
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Dataset
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
# Avaliação
# =====================
def evaluate():
    print(f"Using device: {DEVICE}")

    df = load_data()

    _, X_val, _, y_val = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    val_encodings = tokenizer(
        list(X_val),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    val_dataset = BertDataset(val_encodings, y_val.values)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    print("\nClassification Report (BERT):")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["Ham", "Spam"],
        )
    )

    # =====================
    # Confusion Matrix
    # =====================
    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Ham", "Spam"],
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — BERT Spam Classification")
    plt.show()


if __name__ == "__main__":
    evaluate()
