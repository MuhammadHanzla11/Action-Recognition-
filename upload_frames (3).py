import os
import shutil
import numpy as np

# ================= CONFIG =================
DATASET_PATH = r"G:\PythonProject3\backend\uploads"  # main dataset folder
DEST_FOLDER = os.path.join(DATASET_PATH, "selected_frames")
TOTAL_FRAMES_PER_CLASS = 50  # number of frames to select per class
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
# =========================================

os.makedirs(DEST_FOLDER, exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    # Gather all image paths recursively in this class folder
    all_images = []
    for root, _, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                all_images.append(os.path.join(root, file))

    if len(all_images) == 0:
        print(f"No images found for class: {class_name}")
        continue

    # Select frames evenly spaced to get TOTAL_FRAMES_PER_CLASS
    if len(all_images) >= TOTAL_FRAMES_PER_CLASS:
        indices = np.linspace(0, len(all_images) - 1, TOTAL_FRAMES_PER_CLASS, dtype=int)
        selected_images = [all_images[i] for i in indices]
    else:
        # If fewer frames than required, duplicate last frame
        selected_images = all_images + [all_images[-1]] * (TOTAL_FRAMES_PER_CLASS - len(all_images))

    # Create class folder in destination
    dest_class_folder = os.path.join(DEST_FOLDER, class_name)
    os.makedirs(dest_class_folder, exist_ok=True)

    # Copy selected frames to destination
    for i, img_path in enumerate(selected_images):
        ext = os.path.splitext(img_path)[1]
        dst_path = os.path.join(dest_class_folder, f"frame_{i}{ext}")
        shutil.copy2(img_path, dst_path)

    print(f"Class '{class_name}': selected {len(selected_images)} frames")

print(f"All selected frames saved in: {DEST_FOLDER}")
