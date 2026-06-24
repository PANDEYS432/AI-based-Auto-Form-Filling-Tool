# Aadhaar Document OCR Processor

<img width="956" height="977" alt="Screenshot 2026-06-24 152524" src="https://github.com/user-attachments/assets/e760ff2f-9e7b-4914-853d-ebf0f6a4287f" />


A web application for automated Aadhaar document processing using YOLOv8 object detection and Tesseract OCR.

## Features

- **Object Detection**: Uses YOLOv8 to detect Aadhaar document fields (Name, Number, Gender, DOB)
- **Text Extraction**: Extracts text from detected fields using Tesseract OCR
- **Web API**: FastAPI-powered REST API for document processing
- **Web Interface**: HTML frontend for easy document upload and processing
- **Confidence Filtering**: Configurable confidence thresholds for detection accuracy
- **Multi-language OCR**: Supports English and Hindi text recognition

## Project Structure

```
├── api.py                 # FastAPI application and endpoints
├── main.py                # Core processing logic with YOLO and OCR
├── ocr.py                 # OCR utilities
├── Extraction.py          # Data extraction logic
├── best.pt                # YOLOv8 trained model
├── static/
│   └── index.html         # Web frontend
├── uploads/               # Temporary storage for processed documents
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Requirements

- Python 3.12+
- Tesseract-OCR (Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- CUDA-compatible GPU (optional, for faster inference)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract-OCR:
   - **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Usage

### Start the API Server

```bash
python api.py
```

The server will start at `http://localhost:8000`

### Access the Web Interface

Open your browser and navigate to: `http://localhost:8000`

### API Endpoints

#### Upload and Process Aadhaar Document

**POST** `/process-aadhaar`

- **Parameter**: `file` (multipart/form-data) - Aadhaar image file
- **Response**: JSON with detected fields and OCR text

Example response:
```json
{
  "status": "success",
  "request_id": "uuid-string",
  "data": [
    {
      "class_name": "AadharName",
      "confidence": 0.95,
      "ocr_text": "JOHN DOE"
    },
    {
      "class_name": "AadharNumber",
      "confidence": 0.92,
      "ocr_text": "1234 5678 9012"
    }
  ]
}
```

## Configuration

Edit `main.py` to customize:

- `CONF_THRESHOLD`: Minimum confidence level (default: 0.65)
- `PADDING`: Extra pixels around detected crop (default: 10)
- `OCR_LANG`: Tesseract language (default: "eng")
- `TARGET_CLASSES`: Fields to detect (AadharName, AadharNumber, AadharGender, AadharDOB)

## Detected Fields

- **AadharName**: Extracted name
- **AadharNumber**: 12-digit Aadhaar number
- **AadharGender**: Gender information
- **AadharDOB**: Date of birth

## Performance Notes

- Small crops are upscaled for better OCR accuracy
- Images are processed with adaptive thresholding for optimal text extraction
- Temporary files are automatically cleaned up after processing

## Troubleshooting

- **Tesseract not found**: Verify installation and update `TESSERACT_CMD` path in `main.py`
- **Low OCR accuracy**: Adjust `PADDING` and image preprocessing parameters
- **Detection failures**: Check image quality and confidence threshold
