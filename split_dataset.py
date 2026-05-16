import os
import shutil
import random

SOURCE_DIR = "brain_tumor_dataset"
DEST_DIR = "dataset"
TRAIN_RATIO = 0.8

for category in ["yes", "no"]:
    source_folder = os.path.join(SOURCE_DIR, category)
    images = os.listdir(source_folder)
    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)

    train_images = images[:train_count]
    test_images = images[train_count:]

    for img in train_images:
        os.makedirs(os.path.join(DEST_DIR, "train", category), exist_ok=True)
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(DEST_DIR, "train", category, img)
        )

    for img in test_images:
        os.makedirs(os.path.join(DEST_DIR, "test", category), exist_ok=True)
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(DEST_DIR, "test", category, img)
        )

print("Dataset Split Completed ✅")
