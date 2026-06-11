from ultralytics import YOLO
import cv2
from PIL import Image
# from IPython.display import display
import os

# Load trained YOLOv8 model
model = YOLO("best.pt")

# Create folder to save crops
save_dir = "crops"
os.makedirs(save_dir, exist_ok=True)

def load_image(path):
    img_bgr = cv2.imread(path)

    # Check if image loaded
    if img_bgr is None:
        raise FileNotFoundError(f"❌ Image not found or path is wrong: {path}")

    # Classes you want
    target_classes = {"AadharName", "AadharNumber", "AadharGender", "AadharDOB"}

    # Confidence threshold (optional)
    CONF_THRESHOLD = 0.65

    # YOLO inference
    results = model(img_bgr)[0]

    # Handle no detections
    if results.boxes is None or len(results.boxes) == 0:
        print("❌ No detections found.")

    else:
        count = 0

        for i, box in enumerate(results.boxes):

            # Class info
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]

            # Filter only required classes
            if class_name not in target_classes:
                continue

            # Confidence
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(f"✅ Detection {count+1}: {class_name} ({conf:.2f})")

            # Clamp coordinates
            h, w = img_bgr.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Crop
            crop_bgr = img_bgr[y1:y2, x1:x2]

            if crop_bgr.size == 0:
                print("⚠️ Skipping empty crop")
                continue

            # Convert BGR → RGB → PIL
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            # Display
            # display(crop_pil)

            # Save
            save_path = os.path.join(save_dir, f"{class_name}_{count}.jpg")
            crop_pil.save(save_path)

            print(f"💾 Saved: {save_path}")

            count += 1

        if count == 0:
            print("⚠️ No target classes found after filtering.")


if __name__ == "__main__":
    # Load image
    img_path = "aadhar.jpeg"
    load_image(img_path)


