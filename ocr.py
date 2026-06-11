import pytesseract
from PIL import Image
import cv2

def extract_text_from_aadhaar(image_path):
    try:
        # Set Tesseract path (Windows)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or path is incorrect")

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Save temp image
        temp_path = 'temp.png'
        cv2.imwrite(temp_path, thresh)

        # OCR (English + Hindi)
        text = pytesseract.image_to_string(
            Image.open(temp_path),
            lang='eng+hin'
        )

        return text

    except Exception as e:
        return f"Error: {str(e)}"