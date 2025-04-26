import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import Generator as AnimeGANGenerator
from models.psp import pSp
from utils.common import tensor2im
from argparse import Namespace

# -------------------------------
# Config
# -------------------------------
E4E_CHECKPOINT = r"pretrained_models/e4e_ffhq_encode.pt"
BOUNDARY_PATH = r"boundaries/boundary_happy.npy"  # change to match emotion
INPUT_DIR = r"data/original_faces"  # Folder where original images are stored
INTENSITY = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Setup Output Paths
# -------------------------------
output_dir = r"data/reconstructed_images"  # Folder for emotion-edited images
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Load e4e model
# -------------------------------
print(" Loading e4e encoder...")
opts = Namespace(
    checkpoint_path=E4E_CHECKPOINT,
    device=DEVICE,
    encoder_type='Encoder4Editing',
    start_from_latent_avg=True,
    input_nc=3,
    n_styles=18,
    stylegan_size=1024,
    is_train=False,
    learn_in_w=False,
    output_size=1024,
    id_lambda=0,
    lpips_lambda=0,
    l2_lambda=1,
    w_discriminator_lambda=0,
    use_w_pool=False,
    w_pool_size=50,
    use_ballholder_loss=False,
    optim_type='adam',
    batch_size=1,
    resize_outputs=False
)
encoder = pSp(opts).to(DEVICE).eval()

# Process All Images in Input Directory
print(f" Processing images in {INPUT_DIR}...")

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        input_image_path = os.path.join(INPUT_DIR, filename)
        print(f" Processing image: {input_image_path}")

        # Preprocess Input Image
        image = Image.open(input_image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # -------------------------------
        # Latent Encoding & Emotion Editing
        # -------------------------------
        print("✨ Performing latent editing...")
        with torch.no_grad():
            _, latent = encoder(img_tensor, return_latents=True)
            boundary = torch.from_numpy(np.load(BOUNDARY_PATH)).float().to(DEVICE)
            for i in range(4, 9):
                latent[:, i, :] += INTENSITY * boundary

            # Decode the edited latent code
            generated, _ = encoder.decoder([latent], input_is_latent=True, randomize_noise=False)
            generated = encoder.face_pool(generated)
            edited_image = tensor2im(generated[0])
            edited_image = edited_image.resize((256, 256), Image.LANCZOS)

        # Save the emotion-edited image with the same name as the original (just the number)
        base_filename = os.path.splitext(filename)[0]  # Get the filename without extension
        emotion_img_path = os.path.join(output_dir, f"{base_filename}.png")  # Save with just the number
        edited_image.save(emotion_img_path)
        print(f" Saved emotion-edited image: {emotion_img_path}")

print(" All emotion-edited images have been processed and saved.")
