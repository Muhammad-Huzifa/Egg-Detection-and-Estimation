import os
import cv2
import torch
import numpy as np
import urllib.request
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_b_01ec64.pth"
if not os.path.exists(sam_checkpoint):
    print("Downloading SAM model...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        sam_checkpoint
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

def bbox_to_mask(image_path, label_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    img_h, img_w = img.shape[:2]
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not os.path.exists(label_path):
        return False
    with open(label_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        x_center, y_center, width, height = map(float, parts[1:5])
        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([x1, y1, x2, y2])[None, :],
            multimask_output=False
        )
        mask = masks[0].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 6:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(contour) < 6:
            continue
        points = contour.reshape(-1, 2)
        normalized = points / [img_w, img_h]
        points_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in normalized])
        new_lines.append(f"{cls} {points_str}\n")
    if len(new_lines) > 0:
        with open(label_path, 'w') as f:
            f.writelines(new_lines)
        return True
    return False

def process_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        if not images_dir.exists():
            continue
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        total = len(image_files)
        for idx, img_path in enumerate(image_files):
            label_path = labels_dir / f"{img_path.stem}.txt"
            bbox_to_mask(img_path, label_path)
            print(f"{split}: {idx+1}/{total}")

if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "Eggs-dpy01-1"
    process_dataset(dataset_path)