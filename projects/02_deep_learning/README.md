# SMS Spam Classification — Deep Learning with PyTorch and Transformers

## 📌 Project Overview

This project presents an end-to-end **Natural Language Processing (NLP)** pipeline for
SMS spam classification, evolving from classical neural baselines to **fine-tuning a
state-of-the-art Transformer model (BERT)**.

The goal is to demonstrate practical Deep Learning skills using **PyTorch** and
**Hugging Face Transformers**, following industry-standard workflows.

---

## 💼 Business Problem

Spam messages negatively impact user experience, increase security risks, and reduce
trust in communication platforms.

An effective spam detection system must prioritize **recall for spam messages**, ensuring
that malicious or unwanted content is detected as early as possible.

---

## 📂 Dataset

-   **Name:** SMS Spam Collection
-   **Source:** UCI Machine Learning Repository
-   **Task:** Binary text classification (Spam vs Ham)

The dataset contains short text messages with a **class imbalance**, making it suitable
for evaluating models using metrics beyond accuracy.

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights obtained during EDA:

-   Spam messages are less frequent than ham messages.
-   Spam messages tend to be longer and contain specific patterns.
-   Class imbalance motivates the use of recall and F1-score as primary metrics.

---

## 🛠️ Data Preprocessing

-   Tokenization using a pretrained **BERT tokenizer**
-   Padding and truncation to a fixed sequence length
-   Label encoding (ham = 0, spam = 1)
-   Custom PyTorch Dataset implementations for efficient batching

---

## 🧠 Model Evolution: Baseline → LSTM → BERT

The project was developed incrementally to reflect real-world ML workflows.

### Baseline Model

-   Embedding layer
-   Global average pooling
-   Fully connected layers

This model served as a fast, interpretable baseline.

### LSTM-Based Model

-   Embedding layer
-   LSTM to capture word order and sequential dependencies
-   Fully connected classification head

The LSTM improved representation of sentence structure and recall for spam messages.

### BERT Fine-Tuned Model

-   Pretrained **BERT (bert-base-uncased)**
-   Fine-tuning of all layers on the spam classification task
-   Transformer-based contextual embeddings

This approach achieved the strongest overall performance and reflects
modern NLP practices used in production systems.

---

## 📊 Model Comparison — Conceptual

| Aspect               | Baseline  | LSTM         | BERT Fine-Tuned |
| -------------------- | --------- | ------------ | --------------- |
| Architecture         | Simple NN | Recurrent NN | Transformer     |
| Sequence Awareness   | ❌ No     | ✅ Yes       | ✅ Yes (Strong) |
| Model Complexity     | Low       | Medium       | High            |
| Training Time        | Fast      | Moderate     | Slower          |
| Context Modeling     | Limited   | Improved     | Excellent       |
| Production Readiness | Baseline  | Intermediate | High            |

---

## 📈 Quantitative Results (Validation Set)

| Metric (Spam Class) | Baseline | LSTM     | BERT          |
| ------------------- | -------- | -------- | ------------- |
| Accuracy            | ~0.86    | ~0.87    | **Highest**   |
| Precision           | Moderate | High     | **Very High** |
| Recall              | Moderate | Improved | **Best**      |
| F1-score            | Baseline | Improved | **Best**      |

Although overall accuracy differences are small, **BERT significantly improves recall
and F1-score for the spam class**, which is the most relevant metric for the business goal.

---

## 🔧 Training Strategy

-   Framework: PyTorch
-   Optimizer:
    -   Adam (Baseline / LSTM)
    -   AdamW (BERT fine-tuning)
-   Loss function: CrossEntropyLoss
-   Device support: CPU and **GPU (CUDA enabled)**

Manual training loops were implemented for full control over optimization and evaluation.

---

## ▶️ How to Run the Project

From the root of the repository:

```bash
# Baseline / LSTM
python -m projects.02_deep_learning.src.train
python -m projects.02_deep_learning.src.evaluate

# BERT Fine-Tuning
python -m projects.02_deep_learning.src.train_bert
python -m projects.02_deep_learning.src.evaluate_bert
```

Dataset must be placed at:
projects/02_deep_learning/data/raw/SMSSpamCollection

---

## 📌 Results and Conclusion

This project demonstrates a clear progression in Natural Language Processing modeling,
starting from a simple neural baseline and advancing to a fine-tuned Transformer model.

The experiments show that:

-   Baseline neural models provide strong and interpretable reference performance.
-   LSTM-based architectures improve the modeling of sequential dependencies in text.
-   Fine-tuning a pretrained BERT model delivers the best overall results, particularly
    in terms of **recall and F1-score for the spam class**, which are critical metrics
    for real-world spam detection systems.

Although overall accuracy differences between models are relatively small, more advanced
architectures significantly reduce false negatives, aligning model performance with
business objectives.

This progressive approach reflects real-world machine learning workflows and highlights
strong practical understanding of Deep Learning, model selection, and evaluation strategies.

---

## 🚀 Next Steps

Potential future improvements include:

-   Hyperparameter optimization for BERT fine-tuning.
-   Experimenting with lighter Transformer architectures for faster inference.
-   Threshold tuning to further optimize recall-sensitive applications.
-   Deploying the model as an inference API or integrating it into a production pipeline.

---

## 👤 Author

José Geraldo do Espírito Santo Júnior  
AI & Machine Learning Portfolio  
Location: Brazil
