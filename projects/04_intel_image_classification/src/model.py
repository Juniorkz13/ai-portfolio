import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, fine_tune: bool = False):
    """
    Cria um modelo ResNet50 com Transfer Learning.

    Args:
        num_classes (int): número de classes do dataset
        fine_tune (bool): se True, libera camadas finais para fine-tuning
    """

    # Carregar modelo pré-treinado no ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Congelar todos os parâmetros
    for param in model.parameters():
        param.requires_grad = False

    # Substituir a camada fully connected
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    # Fine-tuning parcial (opcional)
    if fine_tune:
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model
