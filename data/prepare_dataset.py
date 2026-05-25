# prepare_dataset.py — Week 7
# Downloads a subset of the COCO dataset for object detection training.
#
# Run: python data/prepare_dataset.py
# Requires: pip install pycocotools requests tqdm

# TODO — Week 7: implement dataset download and preprocessing
# Steps:
#   1. Download COCO 2017 train/val images (or a subset by category)
#   2. Download COCO annotations JSON
#   3. Filter to your chosen classes (start with 5-10)
#   4. Resize all images to 416x416
#   5. Convert annotations to YOLO format (x_center, y_center, width, height — normalized)
#   6. Split into train/val/test folders

CLASSES = [
    "person", "car", "bicycle", "dog", "cat"
    # Add more from COCO's 80 classes as needed
]

print("Dataset preparation — implement in Week 7")
print(f"Target classes: {CLASSES}")
