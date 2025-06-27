import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2

# Settings
PATCH_SIZE = 512
OVERLAP = 0  # If you want overlap, set this to something like PATCH_SIZE // 2
input_folder = 'dataset/elevation'     # TMC images folder
output_folder = 'patched/highresoelevate'    # Where enhanced JPGs go
os.makedirs(output_folder, exist_ok=True)

def enhance_contrast(img):
    """
    Enhance contrast using CLAHE (adaptive histogram equalization).
    img: grayscale or 3-channel numpy array (uint8)
    returns: contrast-enhanced uint8 image
    """
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(img)
    elif len(img.shape) == 3:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        channels = cv2.split(img)
        eq_channels = [clahe.apply(c) for c in channels]
        return cv2.merge(eq_channels)

def patch_tif(filepath, filename_prefix):
    with rasterio.open(filepath) as src:
        width = src.width
        height = src.height
        count = 0

        for top in tqdm(range(0, height, PATCH_SIZE)):
            for left in range(0, width, PATCH_SIZE):
                window = Window(left, top, PATCH_SIZE, PATCH_SIZE)

                try:
                    patch = src.read(window=window)
                except:
                    continue

                # Convert from (C, H, W) to (H, W, C)
                if patch.shape[0] == 1:
                    patch = patch[0]
                else:
                    patch = patch.transpose(1, 2, 0)

                # Stretch contrast using percentiles
                p2, p98 = np.percentile(patch, 2), np.percentile(patch, 98)
                patch = np.clip((patch - p2) / (p98 - p2 + 1e-5), 0, 1)

                # Scale to 0-255
                patch_uint8 = np.uint8(patch * 255)

                # If grayscale, add channel
                if len(patch_uint8.shape) == 2:
                    patch_uint8 = np.stack([patch_uint8] * 3, axis=-1)

                # Apply adaptive contrast enhancement (CLAHE)
                enhanced = enhance_contrast(patch_uint8)

                # Save patch
                out_path = os.path.join(output_folder, f"{filename_prefix}_{count}.jpg")
                Image.fromarray(enhanced).save(out_path)
                count += 1

    return count

# Process all .tif files in input folder
for tif_file in os.listdir(input_folder):
    if tif_file.endswith(".tif"):
        filepath = os.path.join(input_folder, tif_file)
        prefix = os.path.splitext(tif_file)[0]
        print(f"📦 Patching and Enhancing {prefix}")
        patch_tif(filepath, prefix)

