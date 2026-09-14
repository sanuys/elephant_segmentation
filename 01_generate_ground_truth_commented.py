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

import os                                    # โมดูลมาตรฐานสำหรับจัดการไฟล์/โฟลเดอร์ (path, listdir, makedirs)
import numpy as np                           # ไลบรารีคำนวณเชิงตัวเลข/อาเรย์ ใช้สร้างและรวม mask
import cv2                                   # OpenCV: ใช้อ่าน-เขียนภาพ, resize, เขียนไฟล์ PNG
from ultralytics import YOLO                 # คลาสสำหรับโหลดและรันโมเดล YOLOv8 (ตัวตรวจจับวัตถุ/instance segmentation)

# Paths are relative to this script's own location, so it works no matter
# where the "elephant_segmentation" project folder is placed on disk.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # หาโฟลเดอร์ที่ไฟล์สคริปต์นี้อยู่ (path แบบเต็ม/absolute)
RAW_DIR = os.path.join(BASE_DIR, "raw_data")             # ต่อ path ไปยังโฟลเดอร์เก็บภาพต้นฉบับ raw_data/
GT_DIR = os.path.join(BASE_DIR, "gt_masks")              # ต่อ path ไปยังโฟลเดอร์ที่จะเก็บ mask ผลลัพธ์ gt_masks/
os.makedirs(GT_DIR, exist_ok=True)                       # สร้างโฟลเดอร์ gt_masks/ ถ้ายังไม่มี (ไม่ error ถ้ามีอยู่แล้ว)

ELEPHANT_CLASS_ID = 20   # COCO class index for "elephant"   # รหัสคลาส "elephant" ในชุดคลาส COCO ที่ YOLO ถูกเทรนมาด้วย
CONF_THRESHOLD = 0.15                                        # ค่าความมั่นใจขั้นต่ำ (confidence) ที่ยอมรับผลตรวจจับของ YOLO

print("Loading YOLOv8x-seg (COCO pretrained)...")   # พิมพ์ข้อความแจ้งสถานะว่ากำลังโหลดโมเดล
detector = YOLO("yolov8x-seg.pt")                   # โหลดไฟล์น้ำหนักโมเดล YOLOv8x-seg เข้ามาเป็นออบเจกต์ตัวตรวจจับ
assert detector.names[ELEPHANT_CLASS_ID] == "elephant"   # ตรวจสอบว่ารหัสคลาส 20 ของโมเดลนี้คือ "elephant" จริง (กันโมเดลผิดเวอร์ชัน)

files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg"))  # ดึงรายชื่อไฟล์ .jpg ทั้งหมดใน raw_data/ แล้วเรียงลำดับตามชื่อ
print(f"Found {len(files)} raw images")   # พิมพ์จำนวนภาพทั้งหมดที่พบ

fallback_needed = []   # ลิสต์เก็บชื่อไฟล์ภาพที่ YOLO หาช้างไม่เจอเลย (จะต้องใช้วิธีสำรองทีหลัง)

for i, fname in enumerate(files, 1):   # วนลูปทีละภาพ พร้อมนับลำดับ i เริ่มจาก 1
    in_path = os.path.join(RAW_DIR, fname)                                        # path เต็มของไฟล์ภาพต้นฉบับ
    out_path = os.path.join(GT_DIR, os.path.splitext(fname)[0] + "_mask.png")      # path เต็มของไฟล์ mask ที่จะบันทึก (ตัดนามสกุล .jpg ออกแล้วต่อ "_mask.png")

    img = cv2.imread(in_path)                              # อ่านไฟล์ภาพเข้ามาเป็นอาเรย์ BGR
    h, w = img.shape[:2]                                   # ดึงความสูง (h) และความกว้าง (w) ของภาพ
    result = detector.predict(img, verbose=False, conf=CONF_THRESHOLD)[0]   # รันโมเดล YOLO ตรวจจับวัตถุในภาพ (ปิด log, ใช้ threshold ที่กำหนด) เอาผลลัพธ์ภาพแรก (ภาพเดียว)

    mask = np.zeros((h, w), dtype=np.uint8)   # สร้าง mask เปล่า (ค่า 0 ทั้งหมด) ขนาดเท่าภาพ ไว้รวมผลลัพธ์
    n_elephants = 0                            # ตัวนับจำนวน instance ของช้างที่เจอในภาพนี้
    if result.masks is not None:   # ถ้าโมเดลตรวจพบวัตถุอย่างน้อย 1 ชิ้น (มี mask ออกมา)
        for seg, cls in zip(result.masks.data.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            # วนลูปพร้อมกันทั้ง mask ของแต่ละวัตถุ (seg) และคลาสที่โมเดลทำนาย (cls) ที่ตรวจพบในภาพนี้
            if int(cls) == ELEPHANT_CLASS_ID:   # ถ้าวัตถุชิ้นนี้ถูกทำนายว่าเป็นคลาส "elephant" เท่านั้น
                seg_resized = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
                # ปรับขนาด mask ของวัตถุนี้ (ซึ่งโมเดลคืนมาเป็นขนาดที่โมเดลใช้ภายใน) ให้เท่ากับขนาดภาพจริง (w, h)
                mask |= (seg_resized > 0.5).astype(np.uint8)
                # แปลง mask ที่ resize แล้วให้เป็นค่า 0/1 (ใช้ threshold 0.5) แล้วรวม (OR บิต) เข้ากับ mask หลัก
                # เพื่อรวมช้างหลายตัวในภาพเดียวกันเข้าด้วยกัน
                n_elephants += 1   # เพิ่มตัวนับจำนวนช้างที่เจอ

    if n_elephants == 0:            # ถ้าภาพนี้ไม่เจอช้างเลยแม้แต่ตัวเดียว
        fallback_needed.append(fname)   # บันทึกชื่อไฟล์นี้ไว้ในลิสต์ เพื่อไปประมวลผลด้วยวิธีสำรองภายหลัง

    binary = mask * 255            # แปลง mask จากค่า 0/1 ให้เป็นค่า 0/255 (ขาว-ดำ) สำหรับบันทึกเป็นไฟล์ภาพ
    cv2.imwrite(out_path, binary)  # บันทึก mask เป็นไฟล์ PNG ที่ path ที่กำหนดไว้

    if i % 10 == 0 or i == len(files):   # ทุก ๆ 10 ภาพ หรือเมื่อถึงภาพสุดท้าย
        print(f"  [{i}/{len(files)}] {fname} -> {n_elephants} elephant instance(s) "
              f"(foreground px = {int((binary > 0).sum())})")
        # พิมพ์ความคืบหน้า: ลำดับภาพ/ทั้งหมด, ชื่อไฟล์, จำนวนช้างที่เจอ, และจำนวนพิกเซล foreground ใน mask

# ---- Fallback pass: generic saliency mask (rembg / U2-Net) for images with 0 detections ----
if fallback_needed:   # ถ้ามีภาพที่ต้องใช้วิธีสำรอง (อย่างน้อย 1 ภาพ)
    print(f"\n{len(fallback_needed)} image(s) had 0 elephant detections, "
          f"falling back to a generic saliency mask (rembg/U2-Net) for these only:")
    # พิมพ์แจ้งจำนวนภาพที่ต้องใช้วิธีสำรอง
    for f in fallback_needed:   # วนลูปพิมพ์รายชื่อภาพเหล่านั้นทีละไฟล์
        print("   -", f)        # พิมพ์ชื่อไฟล์ เพื่อให้ตรวจสอบย้อนหลังได้ง่าย

    from PIL import Image                      # นำเข้าไลบรารี PIL สำหรับเปิด/แปลงรูปภาพ (ใช้เฉพาะส่วนนี้)
    from rembg import remove, new_session      # นำเข้าฟังก์ชัน remove (ตัดพื้นหลัง) และ new_session (โหลดโมเดล) จาก rembg
    session = new_session("u2net")             # สร้าง session ของโมเดล U2-Net (โมเดล saliency แบบทั่วไป ไม่เจาะจงคลาส)

    for fname in fallback_needed:   # วนลูปเฉพาะภาพที่อยู่ในลิสต์ fallback
        in_path = os.path.join(RAW_DIR, fname)                                   # path ของไฟล์ภาพต้นฉบับ
        out_path = os.path.join(GT_DIR, os.path.splitext(fname)[0] + "_mask.png")  # path ของไฟล์ mask ที่จะเขียนทับ/บันทึกใหม่
        img = Image.open(in_path).convert("RGB")   # เปิดภาพด้วย PIL แล้วแปลงเป็นโหมดสี RGB
        result = remove(img, session=session, only_mask=True, post_process_mask=True)
        # เรียกใช้ rembg ตัดพื้นหลังออก โดยขอผลลัพธ์เป็น "เฉพาะ mask" (only_mask=True)
        # และให้ทำ post-process มาส์กให้เนียนขึ้น (post_process_mask=True)
        binary = (np.array(result) > 127).astype(np.uint8) * 255
        # แปลงผลลัพธ์ที่ได้ (ภาพ grayscale) เป็นอาเรย์ numpy, threshold ที่ 127 ให้เป็น mask ไบนารี แล้วคูณ 255 ให้เป็นขาว-ดำ
        cv2.imwrite(out_path, binary)   # บันทึก mask ที่ได้จากวิธีสำรองนี้ทับไฟล์เดิม (ซึ่งก่อนหน้านี้เป็น mask ว่างเปล่าจากลูปแรก)

print("\nDone. Ground-truth masks saved to", GT_DIR)   # พิมพ์แจ้งว่าทำงานเสร็จสิ้น พร้อมบอกโฟลเดอร์ปลายทางที่บันทึก mask ทั้งหมด
if fallback_needed:   # ถ้ามีการใช้วิธีสำรองเกิดขึ้น
    print("NOTE: please spot-check the fallback images listed above, since they used the")
    print("      generic (non elephant-specific) saliency method instead of YOLO detection.")
    # เตือนให้ผู้ใช้ตรวจสอบภาพที่ใช้วิธีสำรองด้วยตาอีกครั้ง เพราะวิธีนี้ไม่เจาะจงว่าวัตถุที่ตัดออกมาคือช้างจริง ๆ
