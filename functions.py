from ultralytics import YOLO
import cv2
from PIL import Image
import pytesseract
import os
import json

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH       = "best.pt"
IMAGE_PATH       = "aadhar.jpeg"
OUTPUT_JSON      = "aadhaar_ocr_results.json"
CONF_THRESHOLD   = 0.65
TESSERACT_CMD    = r'C:\Program Files\Tesseract-OCR\tesseract.exe'   # Windows; remove on Linux/Mac
OCR_LANG         = "eng" #eng+hin
PADDING          = 10   # ← extra pixels added on all 4 sides of every crop

TARGET_CLASSES   = {"AadharName", "AadharNumber", "AadharGender", "AadharDOB"}

# ─── Setup ────────────────────────────────────────────────────────────────────

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ─── OCR Helper ───────────────────────────────────────────────────────────────

def run_ocr(image_path: str) -> str:
    """
    Load a saved crop, preprocess, and return OCR text.
    """
    img = cv2.imread(image_path)
    if img is None:
        return ""

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale small crops so Tesseract has enough resolution
    h, w  = gray.shape
    scale = max(1, 200 // min(h, w) if min(h, w) < 200 else 1)
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)

    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 6"
    text   = pytesseract.image_to_string(Image.fromarray(thresh),
                                         lang=OCR_LANG, config=config)
    return text.strip()

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def process_aadhaar(img_path: str, save_dir: str) -> list[dict]:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_h, img_w = img_bgr.shape[:2]

    model   = YOLO(MODEL_PATH)
    results = model(img_bgr)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print("❌ No detections found.")
        return []

    output_records = []
    count          = 0

    for box in results.boxes:
        cls_id     = int(box.cls[0])
        class_name = results.names[cls_id]

        if class_name not in TARGET_CLASSES:
            continue

        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        # ── Raw bounding box ──────────────────────────────────────────────
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ── Add padding on all sides, clamped to image bounds ─────────────
        x1 = max(0,     x1 - PADDING)
        y1 = max(0,     y1 - PADDING)
        x2 = min(img_w, x2 + PADDING)
        y2 = min(img_h, y2 + PADDING)

        crop_bgr = img_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            print(f"⚠️  Empty crop for {class_name}, skipping.")
            continue

        # ── Save crop ─────────────────────────────────────────────────────
        crop_rgb      = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_filename = f"{class_name}_{count}.jpg"
        crop_path     = os.path.join(save_dir, crop_filename)
        Image.fromarray(crop_rgb).save(crop_path)

        print(f"✅ [{count+1}] {class_name} (conf={conf:.2f}) → {crop_path}")

        # ── OCR ───────────────────────────────────────────────────────────
        ocr_text = run_ocr(crop_path)
        print(f"   📝 OCR: {repr(ocr_text)}")

        output_records.append({
            "class_name":      class_name,
            "confidence":      round(conf, 4),
            "ocr_text":        ocr_text,
        })

        count += 1

    if count == 0:
        print("⚠️  No target classes found after filtering.")

    return output_records


# def main():
#     print(f"🔍 Processing: {IMAGE_PATH}\n")
#     records = process_aadhaar(IMAGE_PATH)

#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(records, f, ensure_ascii=False, indent=2)

#     print(f"\n💾 JSON saved to: {OUTPUT_JSON}")
#     print(json.dumps(records, ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()