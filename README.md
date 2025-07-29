# Anime Style Portrait Generator 🎭✨

A full-stack AI-based web application that transforms real human portraits into anime-style images while allowing **emotion manipulation** such as smiling, sadness, or anger — using **Generative Adversarial Networks (GANs)**.

## Features

- Real-time image-to-anime conversion
- Emotion-aware facial edits (happy, sad, disgust, etc.)
- Latent editing with e4e encoder in StyleGAN2’s W+ space
- Stylization using AnimeGANv2
- Web-based UI (React + Chakra UI) with Flask backend
- Optionally deployable on Telegram bot

## Dataset

- **FFHQ (Flickr-Faces-HQ)** - 70,000 aligned high-res face images (1024x1024)
- Preprocessing: center-cropping, resizing to 256x256, MediaPipe face mesh alignment
- Pseudo-labeled emotions for boundary vector generation

## Architecture

1. **Encoding**: Image encoded to W+ space using e4e
2. **Emotion Manipulation**: Add scaled emotion vectors (e.g., smile, anger)
3. **Decoding**: StyleGAN2 reconstructs edited photo
4. **Stylization**: AnimeGANv2 converts it into anime-style portrait

## Model Training

- Emotion boundaries precomputed from classifiers
- AnimeGANv2 pretrained on real→anime image pairs
- Fine-tuning with adversarial, style, content, and total variation losses
- Trained on RTX 3090 with 8GB VRAM

## Evaluation Metrics

| Metric | Original ↔ Emotion | Emotion ↔ Anime | Original ↔ Anime |
|--------|--------------------|------------------|------------------|
| LPIPS  | 0.1905             | 0.1854           | 0.3006           |
| SSIM   | 0.5343             | 0.6843           | 0.4624           |
| PSNR   | 19.2873            | 18.9537          | 16.1684          |
| IS     | —                  | 2.6724           | —                |

## Sample Outputs

| Emotion | Intensity | Original | Emotion-Edited | Anime-Stylized |
|--------|-----------|----------|----------------|----------------|
| Happy  | 10        | ![Preview](github_readme/og1.jpg) | ![](samples/happy_emotion.png) | ![](samples/happy_anime.png) |
| Disgust| 10        | ![](samples/disgust_input.png) | ![](samples/disgust_emotion.png) | ![](samples/disgust_anime.png) |

## Getting Started

### Clone & Install

```bash
git clone https://github.com/vedantulhe12/animegan_final.git
cd animegan_final
pip install -r requirements.txt
```

### Run Backend

```bash
cd backend
python app.py
```

### Run Frontend

```bash
cd frontend
npm install
npm start
```

## 🌍 Applications

- 🎮 Game avatars with custom expressions
- 📱 Social media filters and AI stickers
- 🎨 Artist tools and character generation
- 🧠 Mental health therapy avatars
- 🧑‍🏫 Anime content creation for comics/courses

## Limitations

- May produce artifacts with extreme edits
- Subtle emotions may get lost during stylization
- Ethical risks: identity spoofing, deepfakes
- Dataset biases possible

## Ethical Guidelines

- Watermark outputs
- Require consent for image inputs
- Inform users about risks of emotional manipulation

## Authors

- Vedant Ulhe (22070126123)  
- Rishith Singh Rawat (22070126088)  
- Samarth Patel (22070126098)  
- Sameer Khatwani (22070126099)  

## References

- [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)
- [StyleGAN2](https://github.com/NVlabs/stylegan2)
- [e4e: Encoder for Editing](https://github.com/omertov/encoder4editing)

## License

MIT License
