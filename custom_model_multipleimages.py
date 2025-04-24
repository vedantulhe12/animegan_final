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

# -------------------------------
# Config
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FOLDER = "sample images"
OUTPUT_FOLDER = "reconstructed_faces"
BOUNDARY_PATH = "boundaries/boundary_happy.npy"  # Change for emotion
E4E_CHECKPOINT = "pretrained_models/e4e_ffhq_encode.pt"
ANIMEGAN_CHECKPOINT = "weights/paprika.pt"
INTENSITY = 1.0

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# Load Models
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
animegan = AnimeGANGenerator().to(DEVICE)
animegan.load_state_dict(torch.load(ANIMEGAN_CHECKPOINT, map_location=DEVICE))
animegan.eval()

# -------------------------------
# Preprocessing Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
boundary = torch.from_numpy(np.load(BOUNDARY_PATH)).float().to(DEVICE)

# -------------------------------
# Process Each Image
# -------------------------------
for fname in os.listdir(INPUT_FOLDER):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    input_path = os.path.join(INPUT_FOLDER, fname)
    output_path = os.path.join(OUTPUT_FOLDER, fname)

    # Load and preprocess
    image = Image.open(input_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, latent = encoder(img_tensor, return_latents=True)
        for i in range(4, 9):
            latent[:, i, :] += INTENSITY * boundary
        generated, _ = encoder.decoder([latent], input_is_latent=True, randomize_noise=False)
        generated = encoder.face_pool(generated)
        edited_image = tensor2im(generated[0]).resize((256, 256), Image.LANCZOS)

    # Stylize with AnimeGAN
    face_tensor = to_tensor(edited_image).unsqueeze(0).to(DEVICE)
    face_tensor = face_tensor * 2 - 1
    with torch.no_grad():
        output = animegan(face_tensor).cpu().squeeze(0).clamp(-1, 1)
        output = output * 0.5 + 0.5
        anime_pil = to_pil_image(output)

    anime_pil.save(output_path)
    print(f"✅ Processed {fname}")
