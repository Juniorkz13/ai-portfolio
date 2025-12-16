# SMS Spam Classification — Deep Learning with PyTorch

## 📌 Project Overview

This project focuses on building a **Deep Learning model for text classification** using **PyTorch**.
The objective is to classify SMS messages as **Spam** or **Ham**, applying an end-to-end NLP pipeline
from exploratory data analysis to model training and evaluation.

The project demonstrates practical usage of neural networks for Natural Language Processing (NLP)
with a production-oriented structure.

---

## 💼 Business Problem

Spam messages represent a significant challenge for communication platforms, impacting user experience
and potentially causing financial loss or security issues.

Automatically detecting spam messages allows companies to:

-   Improve user experience
-   Reduce fraud and phishing risks
-   Optimize moderation and filtering systems

---

## 📂 Dataset

-   **Name:** SMS Spam Collection
-   **Source:** UCI Machine Learning Repository
-   **Task:** Binary text classification (Spam vs Ham)

The dataset consists of short text messages and presents a moderate class imbalance,
making it suitable for discussing evaluation metrics beyond accuracy.

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from EDA include:

-   The dataset is imbalanced, with fewer spam messages.
-   Spam messages tend to be longer than ham messages.
-   Text length distribution helps guide model design decisions.

EDA provided essential insights for preprocessing and modeling strategy.

---

## 🛠️ Data Preprocessing

The preprocessing pipeline includes:

-   Tokenization using a pretrained BERT tokenizer
-   Truncation and padding to a fixed maximum sequence length
-   Label encoding (ham = 0, spam = 1)

A custom **PyTorch Dataset** was implemented to ensure compatibility with DataLoader
and efficient batch processing.

---

## 🤖 Model Architecture

The neural network architecture consists of:

-   Embedding layer
-   Global average pooling
-   Fully connected hidden layer with ReLU activation
-   Dropout for regularization
-   Output layer for binary classification

This simple yet effective architecture allows clear interpretation
and serves as a strong baseline for NLP tasks.

---

## 🔧 Training Strategy

-   Framework: PyTorch
-   Loss function: CrossEntropyLoss
-   Optimizer: Adam
-   Batch size: 32
-   Training epochs: 5
-   Device support: CPU / GPU (if available)

A manual training loop was implemented, providing full control
over forward pass, backpropagation, and validation.

---

## 📈 Model Evaluation

The model was evaluated using:

-   Precision, Recall, and F1-score
-   Confusion Matrix

Special attention was given to **Recall for the Spam class**, aiming to reduce
false negatives and improve practical effectiveness.

---

## 🧪 Technologies Used

-   Python
-   PyTorch
-   Hugging Face Transformers (Tokenizer)
-   Pandas, NumPy
-   Scikit-learn
-   Matplotlib, Seaborn
-   TQDM

---

## ▶️ How to Run the Project

From the root of the repository, run:

```bash
python -m projects.02_deep_learning.src.train
python -m projects.02_deep_learning.src.evaluate
```

---

## 📊 Results and Conclusion

The trained Deep Learning model achieved solid performance in the SMS spam classification task,
demonstrating that even a lightweight neural architecture can effectively handle short-text NLP problems.

Key outcomes of the project include:

-   Consistent reduction in training loss across epochs.
-   Balanced precision and recall, with particular focus on recall for the spam class.
-   Clear separation between spam and ham messages, as shown by the confusion matrix.

This project highlights the importance of:

-   Proper dataset handling when working with PyTorch and DataLoader.
-   Building custom datasets compatible with tokenized inputs.
-   Implementing manual training and evaluation loops for full control over the learning process.

Overall, the solution provides a strong and interpretable baseline for text classification tasks
and can be easily extended to more advanced architectures such as LSTM, GRU, or Transformer-based models.

---

## 🚀 Next Steps

Possible future improvements for this project include:

-   Replacing the current architecture with recurrent models such as LSTM or GRU.
-   Fine-tuning a pretrained Transformer model (e.g., BERT) for improved performance.
-   Applying hyperparameter optimization techniques.
-   Implementing cost-sensitive learning to further reduce false negatives.
-   Deploying the model as an inference API or integrating it into a production pipeline.

---

## 👤 Author

José Geraldo do Espírito Santo Júnior  
AI & Machine Learning Portfolio  
Location: Brazil
