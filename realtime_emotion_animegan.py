import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision import transforms
from argparse import Namespace
from models.psp import pSp
from utils.common import tensor2im
from model import Generator as AnimeGANGenerator

# ------------------- CONFIG -------------------
E4E_CHECKPOINT = "pretrained_models/e4e_ffhq_encode.pt"
ANIMEGAN_CHECKPOINT = "weights/paprika.pt"
BOUNDARY_PATH = "boundaries/boundary_happy.npy"  # replace as needed
EMOTION_INTENSITY = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE = 256  # for output comparison

# ------------------- MODELS -------------------
# Load e4e
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

# Load AnimeGAN
animegan = AnimeGANGenerator().to(DEVICE)
animegan.load_state_dict(torch.load(ANIMEGAN_CHECKPOINT, map_location=DEVICE))
animegan.eval()

# Load boundary
boundary = torch.from_numpy(np.load(BOUNDARY_PATH)).float().to(DEVICE)

# ------------------- TRANSFORMS -------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------- MAIN LOOP -------------------
cap = cv2.VideoCapture(0)
print(" Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to capture frame.")
        break

    # Convert OpenCV BGR to PIL RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Preprocess
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Encode to latent (no manipulation)
        _, latent = encoder(img_tensor, return_latents=True)

        # Decode directly (no boundary)
        generated, _ = encoder.decoder([latent], input_is_latent=True, randomize_noise=False)
        generated = encoder.face_pool(generated)
        reconstructed_img = tensor2im(generated[0])
        reconstructed_pil = reconstructed_img.resize((RESIZE, RESIZE), Image.LANCZOS)

    # Display original and reconstructed
    display = np.hstack([
        cv2.resize(frame, (RESIZE, RESIZE)),
        cv2.cvtColor(np.array(reconstructed_pil), cv2.COLOR_RGB2BGR)
    ])

    cv2.imshow(" Original | 🔁 Reconstructed", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")
