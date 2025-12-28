import os
import shutil
import random

SOURCE_DIR = "data/webcam_gestures"
OUTPUT_DIR = "data/split_dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)

    # Ignore non-directories (safety)
    if not os.path.isdir(cls_path):
        continue

    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, split_images in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

print("Dataset split completed.")
