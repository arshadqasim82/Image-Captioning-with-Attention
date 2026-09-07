from src.model.encoder import CNNEncoder
import os
import torch
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def precompute_features(image_dir, output_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    encoder = CNNEncoder().to(device)
    encoder.eval()

    os.makedirs(output_dir, exist_ok=True)

    image_paths = [os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if Path(f).suffix.lower() in {'.jpg', '.jpeg'}]
    print(
        f"Found {len(image_paths)} images. Starting extraction (this may take a bit)...")

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            features = encoder(image_tensor)

            features = features.squeeze(0).cpu()
            file_name = f"{Path(img_path).stem}.pt"
            save_path = os.path.join(output_dir, file_name)
            torch.save(features, save_path)

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")

    print("Feature extraction complete! Tensors saved to disk.")


if __name__ == "__main__":
    image_dir = os.path.join(project_root, "data", "jpg")
    output_dir = os.path.join(project_root, "data", "features")

    precompute_features(image_dir, output_dir)
