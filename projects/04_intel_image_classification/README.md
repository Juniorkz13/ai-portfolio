# 🖼️ Image Classification with Transfer Learning using PyTorch

## 📌 Project Overview

This project implements a **complete Computer Vision pipeline** for **natural scene image classification** using **Deep Learning and Transfer Learning** with PyTorch.

The objective is to demonstrate **end-to-end ML engineering skills**, including data preparation, GPU training, evaluation with proper metrics, and clean, modular code organization suitable for real-world applications.

---

## 🧠 Problem Description

Given an image of a natural scene, the model predicts one of the following classes:

- buildings  
- forest  
- glacier  
- mountain  
- sea  
- street  

This is a **multi-class image classification** problem with visually similar categories (e.g., glacier vs mountain), which makes it a strong benchmark for generalization.

---

## 📂 Dataset

**Dataset:** Intel Image Classification (Natural Scenes)  
**Source:** Kaggle

### Structure
```
data/raw/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
├── seg_test/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_pred/
```

- `seg_train`: training images  
- `seg_test`: labeled test images  
- `seg_pred`: unlabeled images for inference  

---

## 🏗️ Project Structure

```
project_04_intel_image_classification/
├── data/
│   └── raw/
├── experiments/
│   └── best_model.pth
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

The project is organized to ensure **reproducibility, readability, and scalability**.

---

## 🧪 Data Pipeline

- Image loading via `torchvision.datasets.ImageFolder`
- Input resolution: **224 × 224**
- Normalization: ImageNet mean and standard deviation
- Data augmentation (training only):
  - Random horizontal flip
  - Random rotation
  - Color jitter

---

## 🧠 Model Architecture

- Backbone: **ResNet-50** (pretrained on ImageNet)
- Strategy: **Transfer Learning**
  - Backbone frozen
  - Custom classifier head

### Classifier Head
- Linear (2048 → 256)
- ReLU
- Dropout (0.5)
- Linear (256 → 6)

---

## ⚙️ Training Setup

- Framework: PyTorch
- Device: GPU (NVIDIA RTX 2060)
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 1e-3
- Scheduler: ReduceLROnPlateau
- Epochs: 10
- Checkpointing: best model saved by validation loss

---

## 📊 Evaluation Results

Evaluation performed on the **seg_test** dataset.

| Class      | Precision | Recall | F1-score |
|-----------|-----------|--------|---------|
| buildings | 0.93 | 0.89 | 0.91 |
| forest   | 1.00 | 0.97 | 0.98 |
| glacier  | 0.86 | 0.84 | 0.85 |
| mountain | 0.86 | 0.83 | 0.84 |
| sea      | 0.91 | 0.97 | 0.94 |
| street   | 0.89 | 0.95 | 0.92 |

**Overall Accuracy:** **91%**

### Observations
- Strong performance on visually distinctive classes.
- Most confusion occurs between *glacier* and *mountain*, which is expected.
- Errors are semantically reasonable, indicating good generalization.

---

## ▶️ How to Run

Activate environment:
```bash
conda activate ai
```

Train the model:
```bash
python src/train.py
```

Evaluate on test set:
```bash
python src/evaluate.py
```

---

## 🚀 Future Improvements

- Fine-tuning deeper layers of ResNet-50
- Grad-CAM for explainability
- Hyperparameter optimization
- Inference pipeline for `seg_pred`
- Deployment as an API or web app

---

## 📌 Key Takeaways

This project demonstrates:
- End-to-end Computer Vision workflow
- Real dataset handling and debugging
- Transfer Learning best practices
- GPU-accelerated training
- Honest and interpretable evaluation

It reflects the workflow of a **Machine Learning / Computer Vision Engineer**, not a tutorial-level experiment.
