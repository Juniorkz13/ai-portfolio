import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.model import build_model


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    return image, image_tensor


def predict():
    # Paths
    images_dir = "data/raw/seg_pred"
    checkpoint_path = "experiments/best_model.pth"
    output_dir = "experiments/predictions"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Class names (same order as training)
    class_names = [
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street"
    ]

    # Load model
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Found {len(image_files)} images for prediction")

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(images_dir, img_name)

            original_img, img_tensor = load_image(img_path)
            img_tensor = img_tensor.to(device)

            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]

            # Plot and save result
            plt.figure(figsize=(5, 5))
            plt.imshow(original_img)
            plt.title(f"Prediction: {predicted_class}")
            plt.axis("off")

            save_path = os.path.join(output_dir, img_name)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            print(f"{img_name} → {predicted_class}")

    print(f"\nPredictions saved in: {output_dir}")


if __name__ == "__main__":
    predict()
