import os
import torch
import torchvision.transforms as T
from lpips import LPIPS
from torchmetrics.image.inception import InceptionScore
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Load image and apply transformations
def load_image(path, transform, device):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0).to(device)

# Transformation for LPIPS/PSNR (keeps values in [0,1])
transform_base = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

device = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_is(generated_dir):
    # Modified transform for Inception Score
    transform_is = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Lambda(lambda x: (x * 255).byte())  # Convert to [0-255] uint8
    ])

    imgs = []
    for f in os.listdir(generated_dir):
        img_path = os.path.join(generated_dir, f)
        img = Image.open(img_path).convert('RGB')
        img = transform_is(img)
        imgs.append(img.unsqueeze(0))

    imgs = torch.cat(imgs, dim=0).to(device)
    is_metric = InceptionScore().to(device)
    is_metric.update(imgs)
    return is_metric.compute()[0].item()


def calculate_lpips(dir1, dir2, transform):
    lpips_fn = LPIPS(net='alex').to(device)
    losses = []

    # Check if the directories contain files
    if len(os.listdir(dir1)) == 0 or len(os.listdir(dir2)) == 0:
        print(f"Warning: One of the directories {dir1} or {dir2} is empty!")
        return None  # Or you could return a default value like 0.0

    for fname in os.listdir(dir1):
        path1 = os.path.join(dir1, fname)
        path2 = os.path.join(dir2, fname)
        if not os.path.exists(path2):
            continue
        img1 = load_image(path1, transform, device)
        img2 = load_image(path2, transform, device)
        loss = lpips_fn(img1, img2)
        losses.append(loss.item())

    if len(losses) == 0:
        print(f"Warning: No matching files between {dir1} and {dir2}!")
        return None  # Or you could return a default value like 0.0

    return sum(losses) / len(losses)


# Load image and apply transformations (unchanged)
def load_image(path, transform, device):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0).to(device)

# Transformation pipelines (unchanged)
transform_base = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

device = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_ssim(dir1, dir2):
    ssim_values = []
    win_size = 7  # Explicit window size specification

    for fname in os.listdir(dir1):
        path1 = os.path.join(dir1, fname)
        path2 = os.path.join(dir2, fname)
        if not os.path.exists(path2):
            continue
            
        # Load and resize images with dimension checks
        img1 = np.array(Image.open(path1).convert('RGB').resize((256, 256)))
        img2 = np.array(Image.open(path2).convert('RGB').resize((256, 256)))

        # Verify image dimensions
        if img1.shape[0] < win_size or img1.shape[1] < win_size:
            raise ValueError(f"Image {fname} is too small for SSIM calculation")

        # Calculate SSIM with updated parameters
        ssim_value, _ = ssim(
            img1, 
            img2, 
            win_size=win_size,
            channel_axis=-1,  # Replaces deprecated multichannel=True
            full=True
        )
        ssim_values.append(ssim_value)

    return sum(ssim_values)/len(ssim_values)

def calculate_psnr(dir1, dir2, transform):
    psnr_values = []

    for fname in os.listdir(dir1):
        path1 = os.path.join(dir1, fname)
        path2 = os.path.join(dir2, fname)
        if not os.path.exists(path2):
            continue
        img1 = load_image(path1, transform, device)
        img2 = load_image(path2, transform, device)

        mse = F.mse_loss(img1, img2)
        psnr_value = 10 * torch.log10(1 / mse)
        psnr_values.append(psnr_value.item())

    return sum(psnr_values)/len(psnr_values)

if __name__ == "__main__":
    print("🔍 Evaluating LPIPS, IS, SSIM, PSNR...")

    transform_lpips = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    # 1. Original vs Emotion-edited
    lpips_orig_emotion = calculate_lpips("data/original_faces", "data/reconstructed_faces", transform_lpips)

    # 2. Emotion-edited vs Anime output
    lpips_emotion_anime = calculate_lpips("data/reconstructed_faces", "data/generated_faces", transform_lpips)

    # 3. Original vs Anime output
    lpips_orig_anime = calculate_lpips("data/original_faces", "data/generated_faces", transform_lpips)

    # 4. Inception Score
    inception_score = calculate_is("data/generated_faces")

    # 5. SSIM Original vs Emotion-edited
    ssim_orig_emotion = calculate_ssim("data/original_faces", "data/reconstructed_faces")  # Removed transform
    
    # 6. SSIM Emotion-edited vs Anime output
    ssim_emotion_anime = calculate_ssim("data/reconstructed_faces", "data/generated_faces")  # Removed transform
    
    # 7. SSIM Original vs Anime output
    ssim_orig_anime = calculate_ssim("data/original_faces", "data/generated_faces")

    # 8. PSNR Original vs Emotion-edited
    psnr_orig_emotion = calculate_psnr("data/original_faces", "data/reconstructed_faces", transform_lpips)

    # 9. PSNR Emotion-edited vs Anime output
    psnr_emotion_anime = calculate_psnr("data/reconstructed_faces", "data/generated_faces", transform_lpips)

    # 10. PSNR Original vs Anime output
    psnr_orig_anime = calculate_psnr("data/original_faces", "data/generated_faces", transform_lpips)

    # Save results to CSV
    with open("evaluation_scores.csv", "w") as f:
        f.write("Metric,Score\n")
        
        # Check for None results
        lpips_orig_emotion = lpips_orig_emotion if lpips_orig_emotion is not None else "N/A"
        lpips_emotion_anime = lpips_emotion_anime if lpips_emotion_anime is not None else "N/A"
        lpips_orig_anime = lpips_orig_anime if lpips_orig_anime is not None else "N/A"
        
        f.write(f"LPIPS Original vs Emotion,{lpips_orig_emotion}\n")
        f.write(f"LPIPS Emotion vs Anime,{lpips_emotion_anime}\n")
        f.write(f"LPIPS Original vs Anime,{lpips_orig_anime}\n")
        f.write(f"Inception Score,{inception_score:.4f}\n")
        f.write(f"SSIM Original vs Emotion,{ssim_orig_emotion:.4f}\n")
        f.write(f"SSIM Emotion vs Anime,{ssim_emotion_anime:.4f}\n")
        f.write(f"SSIM Original vs Anime,{ssim_orig_anime:.4f}\n")
        f.write(f"PSNR Original vs Emotion,{psnr_orig_emotion:.4f}\n")
        f.write(f"PSNR Emotion vs Anime,{psnr_emotion_anime:.4f}\n")
        f.write(f"PSNR Original vs Anime,{psnr_orig_anime:.4f}\n")

    print("✅ All metrics written to evaluation_scores.csv")
