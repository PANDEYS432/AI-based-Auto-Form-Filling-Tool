from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import shutil
import os
import uuid

from functions import process_aadhaar   # <-- change this

app = FastAPI()

BASE_UPLOAD_DIR = "uploads"
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the HTML frontend"""
    return FileResponse("static/index.html")

@app.post("/process-aadhaar")
async def process_aadhaar_api(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    request_dir = os.path.join(BASE_UPLOAD_DIR, request_id)

    try:
        # Create UUID folder
        os.makedirs(request_dir, exist_ok=True)

        # Save uploaded file inside UUID folder
        file_path = os.path.join(request_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"🔍 Processing: {file_path}\n")

        # Process file
        records = process_aadhaar(file_path, request_dir)

        print(json.dumps(records, ensure_ascii=False, indent=2))

        return {
            "status": "success",
            "request_id": request_id,
            "data": records
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        if os.path.exists(request_dir):
            shutil.rmtree(request_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)