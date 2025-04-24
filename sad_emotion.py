import numpy as np
import os

# -------------------------------
# Paths
# -------------------------------
BOUNDARY_DIR = "boundaries"
os.makedirs(BOUNDARY_DIR, exist_ok=True)

HAPPY_PATH = os.path.join(BOUNDARY_DIR, "boundary_happy.npy")
ANGRY_PATH = os.path.join(BOUNDARY_DIR, "boundary_angry.npy")
DISGUST_PATH = os.path.join(BOUNDARY_DIR, "boundary_disgust.npy")
SAD_OUTPUT_PATH = os.path.join(BOUNDARY_DIR, "boundary_sad.npy")

# -------------------------------
# Load Emotion Boundaries
# -------------------------------
try:
    happy = np.load(HAPPY_PATH)
    angry = np.load(ANGRY_PATH)
    disgust = np.load(DISGUST_PATH)
    print("✅ Successfully loaded happy, angry, and disgust boundaries.")
except FileNotFoundError as e:
    print("❌ Error loading boundaries:", e)
    exit(1)

# -------------------------------
# Create Sad Boundary
# -------------------------------
# You can experiment with these weights
alpha = -1.0  # reverse happy
beta = 0.6    # include anger
gamma = 0.4   # include disgust

boundary_sad = alpha * happy + beta * angry + gamma * disgust

# Normalize the boundary (optional but recommended)
norm = np.linalg.norm(boundary_sad)
if norm != 0:
    boundary_sad /= norm

# -------------------------------
# Save to .npy
# -------------------------------
np.save(SAD_OUTPUT_PATH, boundary_sad)
print(f"✅ Saved boundary_sad.npy at: {SAD_OUTPUT_PATH}")
