from PIL import Image
import os

input_dir = 'patched/highresoelevate'
threshold = 10  # pixel brightness threshold
keep_ratio = 0.1  # keep only if >10% pixels are above threshold

deleted = 0
total = 0

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        path = os.path.join(input_dir, filename)
        img = Image.open(path).convert('L')  # grayscale
        pixels = list(img.getdata())
        total_pixels = len(pixels)
        non_black_pixels = sum(1 for p in pixels if p > threshold)

        if (non_black_pixels / total_pixels) < keep_ratio:
            os.remove(path)
            deleted += 1
        total += 1

print(f"Deleted {deleted} of {total} images ({deleted/total:.2%})")
