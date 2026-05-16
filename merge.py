import os
import shutil
import random

source = "dataset_binary"
target = "dataset_final"

classes = ["tumor", "notumor"]
split_ratio = 0.8

for cls in classes:

    src_folder = os.path.join(source, cls)

    images = os.listdir(src_folder)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    train_dir = os.path.join(target, "Training", cls)
    test_dir = os.path.join(target, "Testing", cls)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(train_dir, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(test_dir, img)
        )

print("Dataset successfully split 80/20!")