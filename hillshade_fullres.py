import numpy as np
import rasterio
from rasterio.windows import Window
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# SETTINGS
PATCH_SIZE = 512
AZIMUTH = 315
ALTITUDE = 45

input_folder = "dataset/elevation"  # Your original DTM .tif files
output_folder = "patched/highreshillshade"
os.makedirs(output_folder, exist_ok=True)

def create_hillshade_patch(elevation_patch):
    elevation_patch = np.nan_to_num(elevation_patch)
    elevation_patch = elevation_patch.astype(np.float32)
    min_val = np.min(elevation_patch)
    max_val = np.max(elevation_patch)
    norm = (elevation_patch - min_val) / (max_val - min_val + 1e-6)

    ls = LightSource(azdeg=AZIMUTH, altdeg=ALTITUDE)
    return ls.shade(norm, cmap=plt.cm.gray, vert_exag=1, blend_mode='overlay')

for tif_file in os.listdir(input_folder):
    if not tif_file.endswith(".tif"):
        continue

    filepath = os.path.join(input_folder, tif_file)
    prefix = os.path.splitext(tif_file)[0]
    print(f"🌕 Generating hillshade patches for {prefix}")

    with rasterio.open(filepath) as src:
        width, height = src.width, src.height
        count = 0

        for top in tqdm(range(0, height, PATCH_SIZE)):
            for left in range(0, width, PATCH_SIZE):
                window = Window(left, top, PATCH_SIZE, PATCH_SIZE)
                try:
                    elevation_patch = src.read(1, window=window)
                except:
                    continue

                if np.all(elevation_patch == 0):
                    continue

                # Generate hillshade
                hillshade_patch = create_hillshade_patch(elevation_patch)

                # Save as high-quality PNG
                out_name = f"{prefix}_{count}.png"
                out_path = os.path.join(output_folder, out_name)
                plt.imsave(out_path, hillshade_patch, dpi=300)  # high-res
                count += 1

