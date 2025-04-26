import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np
from model import Generator as AnimeGANGenerator
from models.psp import pSp
from utils.common import tensor2im
from argparse import Namespace
import datetime

# -------------------------------
# Config
# -------------------------------
E4E_CHECKPOINT = r"pretrained_models/e4e_ffhq_encode.pt"
ANIMEGAN_CHECKPOINT = r"weights/paprika.pt"
BOUNDARY_PATH = r"boundaries\pggan_celebahq_age_boundary.npy"
INPUT_IMAGE = r"C:/Users/Vedant/Desktop/animegan2-pytorch/data/original_faces/6999.png"
INTENSITY = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

output_dir = "results"
debug_dir = "debug"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

existing_files = [f for f in os.listdir(output_dir) if f.startswith("outimg-") and f.endswith(".png")]
nums = [int(f.split("-")[1].split(".")[0]) for f in existing_files if "-" in f and f.split("-")[1].split(".")[0].isdigit()]
next_num = max(nums) + 1 if nums else 1
OUTPUT_PATH = os.path.join(output_dir, f"outimg-{next_num}.png")

# -------------------------------
# Load e4e model
# -------------------------------
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

# -------------------------------
# Preprocess Image
# -------------------------------
image = Image.open(INPUT_IMAGE).convert("RGB")
print(f"📸 Input Image Path: {INPUT_IMAGE}")

# Save original resized input for comparison
image_resized = image.resize((256, 256), Image.LANCZOS)
image_resized.save(os.path.join(debug_dir, "input_resized.png"))

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
img_tensor = transform(image).unsqueeze(0).to(DEVICE)

# -------------------------------
# Latent Encoding & Editing
# -------------------------------
with torch.no_grad():
    _, latent = encoder(img_tensor, return_latents=True)

    boundary = torch.from_numpy(np.load(BOUNDARY_PATH)).float().to(DEVICE)
    print(f"📐 Boundary norm: {boundary.norm().item():.4f}")

    for i in range(4, 9):
        latent[:, i, :] += INTENSITY * boundary

    print(f"🧬 Latent stats: mean={latent.mean().item():.4f}, std={latent.std().item():.4f}")

    generated, _ = encoder.decoder([latent], input_is_latent=True, randomize_noise=False)
    generated = encoder.face_pool(generated)  # ensures 256x256
    edited_image = tensor2im(generated[0])
    edited_image = edited_image.resize((256, 256), Image.LANCZOS)

    # Save edited face before stylization
    edited_image.save(os.path.join(debug_dir, "before_anime.png"))

# -------------------------------
# Stylize with AnimeGANv2
# -------------------------------
animegan = AnimeGANGenerator().to(DEVICE)
animegan.load_state_dict(torch.load(ANIMEGAN_CHECKPOINT, map_location=DEVICE))
animegan.eval()

face_tensor = to_tensor(edited_image).unsqueeze(0).to(DEVICE)
face_tensor = face_tensor * 2 - 1  # match AnimeGAN input range [-1, 1]

with torch.no_grad():
    output = animegan(face_tensor).cpu().squeeze(0).clamp(-1, 1)
    output = output * 0.5 + 0.5
    anime_pil = to_pil_image(output)

    # Save stylized anime image separately for debug
    anime_pil.save(os.path.join(debug_dir, "after_anime.png"))

# -------------------------------
# Save Output
# -------------------------------
anime_pil.save(OUTPUT_PATH)
print(f"Saved stylized image with emotion at: {OUTPUT_PATH}")


