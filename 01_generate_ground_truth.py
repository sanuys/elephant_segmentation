"""
Step 1: Generate ground-truth masks for the elephant segmentation dataset.

The raw_data folder only contains 56 JPG photographs of elephants; it does
not include any hand-made segmentation masks. Manually drawing pixel-
accurate masks for 56 photos is not practical here, so — as allowed by the
workshop instructions ("ถ้าไม่มี ให้สร้างขึ้นมาเอง") — we generate a
reference mask automatically.

Method (v2 - CLASS-SPECIFIC, elephants only)
----------------------------------------------
We use YOLOv8x-seg, an instance-segmentation model pretrained on COCO
(which includes an "elephant" class). For every image we run the detector
and keep ONLY the instance masks whose predicted class is "elephant"
(COCO class id 20) — union them if there is more than one elephant in the
photo. This means people, vehicles, other animals, and background are
never included in the mask, unlike a generic class-agnostic saliency
model (which is what version 1 of this script used, via `rembg`/U2-Net,
and which sometimes also picked up a nearby human handler).

Fallback: a small number of images have an unusual angle/lighting where
YOLO fails to recognise the elephant at all (0 detections). For those few
images only, we fall back to the generic U2-Net saliency mask (`rembg`) so
every image still gets *some* reasonable reference mask. These fallback
images are printed at the end of the run so they can be spot-checked.

IMPORTANT (for the report / discussion section):
These masks are a "pseudo ground truth" produced by pretrained models, not
hand-verified by a person. Good enough as a reference for evaluating our
CLASSICAL (non deep-learning) segmentation algorithm with a confusion
matrix / ROC curve, but this should be disclosed as a limitation.
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO

# Paths are relative to this script's own location, so it works no matter
# where the "elephant_segmentation" project folder is placed on disk.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
GT_DIR = os.path.join(BASE_DIR, "gt_masks")
os.makedirs(GT_DIR, exist_ok=True)

ELEPHANT_CLASS_ID = 20   # COCO class index for "elephant"
CONF_THRESHOLD = 0.15

print("Loading YOLOv8x-seg (COCO pretrained)...")
detector = YOLO("yolov8x-seg.pt")
assert detector.names[ELEPHANT_CLASS_ID] == "elephant"

files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg"))
print(f"Found {len(files)} raw images")

fallback_needed = []

for i, fname in enumerate(files, 1):
    in_path = os.path.join(RAW_DIR, fname)
    out_path = os.path.join(GT_DIR, os.path.splitext(fname)[0] + "_mask.png")

    img = cv2.imread(in_path)
    h, w = img.shape[:2]
    result = detector.predict(img, verbose=False, conf=CONF_THRESHOLD)[0]

    mask = np.zeros((h, w), dtype=np.uint8)
    n_elephants = 0
    if result.masks is not None:
        for seg, cls in zip(result.masks.data.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            if int(cls) == ELEPHANT_CLASS_ID:
                seg_resized = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
                mask |= (seg_resized > 0.5).astype(np.uint8)
                n_elephants += 1

    if n_elephants == 0:
        fallback_needed.append(fname)

    binary = mask * 255
    cv2.imwrite(out_path, binary)

    if i % 10 == 0 or i == len(files):
        print(f"  [{i}/{len(files)}] {fname} -> {n_elephants} elephant instance(s) "
              f"(foreground px = {int((binary > 0).sum())})")

# ---- Fallback pass: generic saliency mask (rembg / U2-Net) for images with 0 detections ----
if fallback_needed:
    print(f"\n{len(fallback_needed)} image(s) had 0 elephant detections, "
          f"falling back to a generic saliency mask (rembg/U2-Net) for these only:")
    for f in fallback_needed:
        print("   -", f)

    from PIL import Image
    from rembg import remove, new_session
    session = new_session("u2net")

    for fname in fallback_needed:
        in_path = os.path.join(RAW_DIR, fname)
        out_path = os.path.join(GT_DIR, os.path.splitext(fname)[0] + "_mask.png")
        img = Image.open(in_path).convert("RGB")
        result = remove(img, session=session, only_mask=True, post_process_mask=True)
        binary = (np.array(result) > 127).astype(np.uint8) * 255
        cv2.imwrite(out_path, binary)

print("\nDone. Ground-truth masks saved to", GT_DIR)
if fallback_needed:
    print("NOTE: please spot-check the fallback images listed above, since they used the")
    print("      generic (non elephant-specific) saliency method instead of YOLO detection.")
