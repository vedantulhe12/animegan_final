import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# === GANimation Generator ===
from net.generator import G_net  # Your PyTorch-converted Generator class

# === Load AU dictionary (basic version) ===
AU_EMOTION_MAP = {
    "happy":  {"AU06": 1.0, "AU12": 1.0},  # cheek raise, smile
    "sad":    {"AU01": 1.0, "AU04": 1.0, "AU15": 1.0},  # inner brow raise, brow lower, lip corner depressor
    "angry":  {"AU04": 1.0, "AU05": 1.0, "AU07": 1.0},  # brow lower, upper lid raise, lid tighten
    "surprise": {"AU01": 1.0, "AU02": 1.0, "AU05": 1.0, "AU26": 1.0},  # brow raise, upper lid raise, jaw drop
}

AU_INDEX = {
    "AU01": 0, "AU02": 1, "AU04": 2, "AU05": 3, "AU06": 4,
    "AU07": 5, "AU10": 6, "AU12": 7, "AU14": 8, "AU15": 9,
    "AU17": 10, "AU23": 11, "AU25": 12, "AU26": 13, "AU45": 14
}

# === Setup paths ===
input_image_path = r"C:\Users\Vedant\Desktop\animegan2-pytorch\inputs images\7acb0a06072b15d99b3989f7df009146_69999.png"  # path to frontal face
output_image_path = r"emotion_output.png"
emotion = "happy"
intensity = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load and preprocess image ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

img = Image.open(input_image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# === Build AU vector ===
au_vector = torch.zeros((1, 15)).to(device)
for au_name, value in AU_EMOTION_MAP[emotion].items():
    index = AU_INDEX[au_name]
    au_vector[0, index] = value * intensity

# === Load generator ===
G = G_net().to(device)
G.load_state_dict(torch.load(r"C:\Users\Vedant\Desktop\GANimation\checkpoints\experiment_1\opt_epoch_30_id_G.pth", map_location=device))  # Path to pretrained GANimation G
G.eval()

# === Forward pass ===
with torch.no_grad():
    fake_img, _, _ = G(img_tensor)

# === Save result ===
output_img = transforms.ToPILImage()(fake_img.squeeze().cpu().clamp(0, 1))
output_img.save(output_image_path)

print(f"Saved emotion-enhanced image to: {output_image_path}")
