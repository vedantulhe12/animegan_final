import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from model import Generator as AnimeGANGenerator
from models.psp import pSp
from utils.common import tensor2im
from argparse import Namespace


class EmotionAnimeGAN(torch.nn.Module):
    def __init__(self, e4e_ckpt, animegan_ckpt, device='cuda'):
        super().__init__()
        self.device = device

        # Load e4e encoder
        opts = Namespace(
            checkpoint_path=e4e_ckpt,
            device=device,
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
        self.encoder = pSp(opts).to(device).eval()

        # Load AnimeGAN
        self.animegan = AnimeGANGenerator().to(device).eval()
        self.animegan.load_state_dict(torch.load(animegan_ckpt, map_location=device))

    def forward(self, input_image: Image.Image, boundary_path: str, intensity: float = 1.0) -> Image.Image:
        # Convert and normalize image
        image_tensor = to_tensor(input_image).unsqueeze(0).to(self.device)
        image_tensor = torch.nn.functional.interpolate(image_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

        # Encode + apply boundary
        with torch.no_grad():
            _, latent = self.encoder(image_tensor, return_latents=True)
            boundary = torch.from_numpy(np.load(boundary_path)).float().to(self.device)
            for i in range(18):
                latent[:, i, :] += intensity * boundary

            generated, _ = self.encoder.decoder([latent], input_is_latent=True, randomize_noise=False)
            generated = self.encoder.face_pool(generated)
            edited_image = tensor2im(generated[0]).resize((256, 256), Image.LANCZOS)

        # Stylize
        with torch.no_grad():
            face_tensor = to_tensor(edited_image).unsqueeze(0).to(self.device)
            face_tensor = face_tensor * 2 - 1  # Normalize to [-1, 1] for AnimeGAN
            output = self.animegan(face_tensor).cpu().squeeze(0).clamp(-1, 1)
            output = output * 0.5 + 0.5  # Back to [0, 1]
            return to_pil_image(output)


# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual model checkpoints and image
    e4e_checkpoint = "pretrained_models/e4e_ffhq_encode.pt"
    animegan_checkpoint = "weights/paprika.pt"
    boundary_path = "boundaries/boundary_happy.npy"
    image_path = "data/original_faces/69048.png"
    output_path = "results/test_wrapped.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    input_image = Image.open(image_path).convert("RGB")

    # Run model
    model = EmotionAnimeGAN(e4e_checkpoint, animegan_checkpoint, device)
    result = model.forward(input_image, boundary_path, intensity=1.0)

    # Save output
    result.save(output_path)
    print(f"Saved stylized image to {output_path}")
