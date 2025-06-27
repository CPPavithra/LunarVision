import os
import numpy as np
import cv2
from tqdm import tqdm

# === Paths ===
tmc_dir = "patched/highresoimages"
dtm_dir = "patched/highresoelevate"
hill_dir = "patched/highreshillshade"

output_npy = "final_dataset/npy"
output_jpg = "final_dataset/jpg"
os.makedirs(output_npy, exist_ok=True)
os.makedirs(output_jpg, exist_ok=True)

# Get TMC filenames
tmc_files = sorted([f for f in os.listdir(tmc_dir) if f.endswith(".jpg")])
matched = 0

for tmc_file in tqdm(tmc_files):
    prefix = os.path.splitext(tmc_file)[0]  # e.g., img1_123
    dtm_path = os.path.join(dtm_dir, prefix + ".jpg")
    hill_path = os.path.join(hill_dir, prefix + ".png")  # 🔧 fixed here

    if not os.path.exists(dtm_path) or not os.path.exists(hill_path):
        continue  # Skip if any modality is missing

    # Load all 3 images in grayscale
    tmc = cv2.imread(os.path.join(tmc_dir, tmc_file), cv2.IMREAD_GRAYSCALE)
    dtm = cv2.imread(dtm_path, cv2.IMREAD_GRAYSCALE)
    hill = cv2.imread(hill_path, cv2.IMREAD_GRAYSCALE)

    # Resize to same shape (precaution)
    h, w = tmc.shape
    dtm = cv2.resize(dtm, (w, h))
    hill = cv2.resize(hill, (w, h))

    # Normalize to [0,1]
    tmc = tmc.astype(np.float32) / 255.0
    dtm = dtm.astype(np.float32) / 255.0
    hill = hill.astype(np.float32) / 255.0

    # Stack [TMC, DTM, Hillshade]
    stacked = np.stack([tmc, dtm, hill], axis=-1)

    # Save .npy
    np.save(os.path.join(output_npy, prefix + ".npy"), stacked)

    # Save .jpg for visual debugging
    vis = (stacked * 255).astype(np.uint8)
    vis_rgb = cv2.merge([vis[:, :, 0], vis[:, :, 1], vis[:, :, 2]])
    cv2.imwrite(os.path.join(output_jpg, prefix + ".jpg"), vis_rgb)

    matched += 1

print(f"\n✅ Done. Total usable samples with all 3 modalities: {matched}")
