import os
import shutil
import random
from collections import defaultdict

# Paths
input_folder = "patched/highresoimages"      # All current 15,000 patches
output_folder = "sampled/images"     # New reduced set (to be created)
os.makedirs(output_folder, exist_ok=True)

# Config
TOTAL_TARGET = 3000  # Total images you want to keep

# Step 1: Group by parent prefix (e.g., img1, img2...)
groups = defaultdict(list)
for fname in os.listdir(input_folder):
    if fname.endswith(".jpg"):
        prefix = fname.split('_')[0]  # 'img1_45.jpg' → 'img1'
        groups[prefix].append(fname)

# Step 2: Sample uniformly
samples_per_group = TOTAL_TARGET // len(groups)
print(f"Sampling about {samples_per_group} from each of {len(groups)} groups")

selected_files = []
for prefix, files in groups.items():
    if len(files) <= samples_per_group:
        selected = files  # Keep all if fewer than needed
    else:
        selected = random.sample(files, samples_per_group)
    selected_files.extend(selected)

# Step 3: Copy to output folder
for fname in selected_files:
    src = os.path.join(input_folder, fname)
    dst = os.path.join(output_folder, fname)
    shutil.copy(src, dst)

print(f"✅ Sampled {len(selected_files)} images into {output_folder}")
