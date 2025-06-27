import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2

# Settings
PATCH_SIZE = 512
input_tmc_folder = 'dataset/images'        # Optical TMC .tif images
input_dtm_folder = 'dataset/elevation'     # Corresponding DTM .tif images
output_folder = 'patched/overlayed'        # Output JPGs
os.makedirs(output_folder, exist_ok=True)

def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    if len(img.shape) == 2:
        return clahe.apply(img)
    elif len(img.shape) == 3:
        channels = cv2.split(img)
        eq_channels = [clahe.apply(c) for c in channels]
        return cv2.merge(eq_channels)

def patch_overlay(tmc_path, dtm_path, filename_prefix):
    with rasterio.open(tmc_path) as tmc, rasterio.open(dtm_path) as dtm:
        width = min(tmc.width, dtm.width)
        height = min(tmc.height, dtm.height)
        count = 0

        for top in tqdm(range(0, height, PATCH_SIZE)):
            for left in range(0, width, PATCH_SIZE):
                window = Window(left, top, PATCH_SIZE, PATCH_SIZE)

                try:
                    tmc_patch = tmc.read(window=window)
                    dtm_patch = dtm.read(window=window)
                except:
                    continue

                # Process TMC patch
                if tmc_patch.shape[0] == 1:
                    tmc_patch = tmc_patch[0]
                else:
                    tmc_patch = tmc_patch.transpose(1, 2, 0)

                # Contrast stretch

                # Enhance TMC image patch
                p2, p98 = np.percentile(tmc_patch, 2), np.percentile(tmc_patch, 98)
                tmc_patch = np.clip((tmc_patch - p2) / (p98 - p2 + 1e-5), 0, 1)
                tmc_uint8 = np.uint8(tmc_patch * 255)

                if len(tmc_uint8.shape) == 2:
                    tmc_uint8 = np.stack([tmc_uint8] * 3, axis=-1)

                # Process DTM patch
                dem = dtm_patch[0]
                dem_norm = np.clip((dem - np.min(dem)) / (np.ptp(dem) + 1e-5), 0, 1)
                dem_uint8 = np.uint8(dem_norm * 255)  # ✅ define before resize

                # Resize DEM to match TMC patch
                dem_uint8_resized = cv2.resize(dem_uint8, (tmc_uint8.shape[1], tmc_uint8.shape[0]), interpolation=cv2.INTER_LINEAR)

                # Apply colormap to resized DEM
                dem_colored = cv2.applyColorMap(dem_uint8_resized, cv2.COLORMAP_JET)

                # Overlay: blend with TMC
                overlay = cv2.addWeighted(tmc_uint8, 0.7, dem_colored, 0.3, 0)

                # Enhance contrast
                enhanced = enhance_contrast(overlay)

                # Save patch
                out_path = os.path.join(output_folder, f"{filename_prefix}_{count}.jpg")
                Image.fromarray(enhanced).save(out_path)
                count += 1

        return count

# Match TMC and DTM files by filename prefix
tmc_files = [f for f in os.listdir(input_tmc_folder) if f.endswith(".tif")]

for tmc_file in tmc_files:
    prefix = os.path.splitext(tmc_file)[0]
    tmc_path = os.path.join(input_tmc_folder, tmc_file)

    # Try to find the matching DTM file
    dtm_path = os.path.join(input_dtm_folder, f"{prefix}.tif")
    if not os.path.exists(dtm_path):
        print(f"⚠️  No matching DTM for {prefix}")
        continue

    print(f"🌀 Overlaying {prefix}")
    patch_overlay(tmc_path, dtm_path, prefix)
