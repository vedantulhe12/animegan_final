from PIL import Image
from wrapper_class import EmotionAnimeGAN
import torch

model = EmotionAnimeGAN(
    e4e_ckpt="pretrained_models/e4e_ffhq_encode.pt",
    animegan_ckpt="weights/paprika.pt",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

input_image = Image.open("data/original_faces/69048.png").convert("RGB")
emotion = "happy"
intensity = 1.0
boundary_path = f"boundaries/boundary_{emotion}.npy"

output = model(input_image, boundary_path, intensity)
output.save("results/test_wrapped.png")
