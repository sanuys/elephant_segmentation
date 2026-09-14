# Elephant Segmentation

เวิร์กชอป **การแบ่งส่วนภาพช้าง (Elephant Segmentation)** ด้วยการประมวลผลภาพเชิงมอร์โฟโลยี
รายวิชา **Digital Image Processing — Lecture 10 (Morphological Processing)**

แบ่งส่วนภาพเพื่อแยก **ช้าง (foreground)** ออกจาก **พื้นหลัง (background)** ด้วยอัลกอริทึมแบบดั้งเดิม (classical, ใช้ระยะห่างสีในปริภูมิ CIE-Lab) แล้วปรับปรุงผลลัพธ์ด้วยตัวดำเนินการเชิงมอร์โฟโลยี (opening, closing, region filling, small-blob removal) จากนั้นประเมินผลด้วย Confusion Matrix และ ROC Curve

## สมาชิกกลุ่ม

| ชื่อ-นามสกุล | รหัสนักศึกษา |
|---|---|
|ชนสรณ์ ควรสุวรรณ | 6710301004 |
| ปิ่นพงศ์ ชูสุวรรณ์ | 6710301023 |
| มูฮัมหมัดฮาซัน อุเซ็ง | 6710301025 |
| คชา สวัสดิพิศาล | 6710301054 |
## โครงสร้างโปรเจกต์

```
elephant_segmentation/
├── raw_data/                          # ภาพถ่ายช้างดิบ 56 ภาพ (จาก Roboflow)
├── gt_masks/                          # ground-truth mask ที่สร้างขึ้นเอง (1 ไฟล์ต่อภาพ)
├── outputs/                           # กราฟ/รูปผลลัพธ์และ metrics_summary.json
├── 01_generate_ground_truth.py        # สคริปต์สร้าง ground-truth mask (YOLOv8x-seg + rembg สำรอง)
├── elephant_segmentation_workshop_TH.ipynb   # โน้ตบุ๊กหลักของเวิร์กชอป (ภาษาไทย)
├── SUMMARY_TH.md                      # สรุปผลโครงการภาษาไทย
└── requirements.txt
```

## วิธีใช้งาน

1. ติดตั้ง dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (ถ้ายังไม่มี `gt_masks/`) สร้าง ground-truth mask ก่อน:
   ```bash
   python 01_generate_ground_truth.py
   ```
3. เปิดและรันโน้ตบุ๊ก `elephant_segmentation_workshop_TH.ipynb` ตามลำดับเซลล์ ผลลัพธ์ (กราฟ, confusion matrix, `metrics_summary.json`) จะถูกบันทึกลงในโฟลเดอร์ `outputs/`

## ขั้นตอนการทำงาน

1. **เลือกงานแบ่งส่วนภาพ** — ช้าง (foreground) เทียบกับพื้นหลัง
2. **ชุดข้อมูล ≥ 50 ภาพ + ground truth** — ภาพ 56 ภาพ; ground truth จาก YOLOv8x-seg (เฉพาะคลาสช้าง) พร้อมวิธีสำรองแบบ saliency (U2-Net) สำหรับภาพที่ตรวจไม่พบช้าง
3. **ออกแบบอัลกอริทึมแบบดั้งเดิม** — threshold จากระยะห่างสี Lab เทียบกับสีขอบภาพ
4. **ปรับปรุงด้วยมอร์โฟโลยี** — opening → closing → เติมรู → ลบก้อนเล็ก
5. **ประเมินผล** — Confusion Matrix และ ROC Curve พร้อมอภิปรายผล
6. **นำเสนอผล** — ดูสรุปฉบับเต็มได้ที่ [`SUMMARY_TH.md`](SUMMARY_TH.md)

## ผลลัพธ์โดยสรุป

| ตัวชี้วัด | Threshold ดิบ | หลังทำมอร์โฟโลยี |
|---|---|---|
| Accuracy | 72.84% | **74.21%** |
| Precision | 63.69% | **66.37%** |
| Recall (TPR) | 57.22% | **57.61%** |
| F1-score | 0.6028 | **0.6168** |
| ROC AUC | — | **0.733** |

รายละเอียดการวิเคราะห์ ข้อจำกัด และแนวทางปรับปรุง อ่านเพิ่มเติมได้ที่ [`SUMMARY_TH.md`](SUMMARY_TH.md)
