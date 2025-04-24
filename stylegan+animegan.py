import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np
from model import Generator as AnimeGANGenerator
from models.psp import pSp
from utils.common import tensor2im
import datetime
from argparse import Namespace

# -------------------------------
# User Input for Emotion and Intensity
# -------------------------------
emotion = input("Enter the emotion (e.g., happy, angry, sad): ").strip().lower()
intensity_input = input("Enter intensity (e.g., 0.5, 1.0, 1.5): ").strip()
try:
    INTENSITY = float(intensity_input)
except ValueError:
    print("Invalid intensity. Defaulting to 1.0.")
    INTENSITY = 1.0

BOUNDARY_PATH = f"boundaries/boundary_{emotion}.npy"
if not os.path.exists(BOUNDARY_PATH):
    raise FileNotFoundError(f"❌ Emotion boundary file not found: {BOUNDARY_PATH}")

# -------------------------------
# Config
# -------------------------------
E4E_CHECKPOINT = r"pretrained_models/e4e_ffhq_encode.pt"
ANIMEGAN_CHECKPOINT = r"weights/celeba_distill.pt"
INPUT_IMAGE = r"C:\Users\Vedant\Desktop\animegan2-pytorch\data\original_faces\69048.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
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
transform = transforms.Compose([
    transforms.Resize((256, 256)),
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
    for i in range(0, 18):
        latent[:, i, :] += INTENSITY * boundary

    generated, _ = encoder.decoder([latent], input_is_latent=True, randomize_noise=False)
    generated = encoder.face_pool(generated)  # ensures 256x256
    edited_image = tensor2im(generated[0])
    edited_image = edited_image.resize((256, 256), Image.LANCZOS)

# -------------------------------
# Stylize with AnimeGANv2
# -------------------------------
animegan = AnimeGANGenerator().to(DEVICE)
animegan.load_state_dict(torch.load(ANIMEGAN_CHECKPOINT, map_location=DEVICE))
animegan.eval()

face_tensor = to_tensor(edited_image).unsqueeze(0).to(DEVICE)
face_tensor = face_tensor * 2 - 1

with torch.no_grad():
    output = animegan(face_tensor).cpu().squeeze(0).clamp(-1, 1)
    output = output * 0.5 + 0.5
    anime_pil = to_pil_image(output)

# -------------------------------
# Save Output Image Only
# -------------------------------
anime_pil.save(OUTPUT_PATH)
print(f"✅ Anime-styled image saved at: {OUTPUT_PATH}")
