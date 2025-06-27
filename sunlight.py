import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from tqdm import tqdm

input_dir = 'patched/highresoelevate'          # your .jpg patches
output_dir = 'hillshaded_patches'
os.makedirs(output_dir, exist_ok=True)

azimuths = [45, 135, 225, 315]         # Simulate different sunlight directions

images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

for az in azimuths:
    az_dir = os.path.join(output_dir, f'az{az}')
    os.makedirs(az_dir, exist_ok=True)
    ls = LightSource(azdeg=az, altdeg=45)

    for img_file in tqdm(images, desc=f"Az {az}"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        elevation = img.astype(np.float32)
        elevation /= 255.0  # Normalize to 0-1

        hillshade = ls.shade(elevation, cmap=plt.cm.gray, vert_exag=1, blend_mode='overlay')

        out_path = os.path.join(az_dir, img_file.replace('.jpg', f'_az{az}.png'))
        plt.imsave(out_path, hillshade, cmap='gray')

