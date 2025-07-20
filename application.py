from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np
import io
import base64
from pathlib import Path
import os
import tempfile
import pickle
from skimage.transform import resize
import asyncio
from typing import Optional
from fastapi.responses import FileResponse, JSONResponse




app = FastAPI(title="Parking Spot Detection System")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Constants
EMPTY = True
NOT_EMPTY = False

# Global variables for model and mask
MODEL = None
MASK = None
PARKING_SPOTS = None

def load_model_and_mask():
    """Load the ML model and parking mask"""
    global MODEL, MASK, PARKING_SPOTS
    try:
        # Load your trained model
        MODEL = pickle.load(open("model/model.p", "rb"))
        
        # Load parking mask
        MASK = cv2.imread("mask_1920_1080.png", 0)
        
        # Get parking spots from mask
        if MASK is not None:
            connected_components = cv2.connectedComponentsWithStats(MASK, 4, cv2.CV_32S)
            PARKING_SPOTS = get_parking_spots_bboxes(connected_components)
            
    except Exception as e:
        print(f"Error loading model or mask: {e}")
        MODEL = None
        MASK = None
        PARKING_SPOTS = None

def empty_or_not(spot_bgr):
    """Determine if parking spot is empty or not"""
    if MODEL is None:
        return EMPTY  # Default to empty if model not loaded
    
    flat_data = []
    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    
    y_output = MODEL.predict(flat_data)
    
    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY

def get_parking_spots_bboxes(connected_components):
    """Extract parking spot bounding boxes from connected components"""
    (totalLabels, label_ids, values, centroid) = connected_components
    
    slots = []
    coef = 1
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
    
    return slots

def calc_diff(im1, im2):
    """Calculate difference between two images"""
    return np.abs(np.mean(im1) - np.mean(im2))

def process_frame(frame, spots, spots_status):
    """Process a frame and draw parking spot rectangles"""
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # Add status text
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    available_spots = sum(spots_status) if spots_status else 0
    total_spots = len(spots_status) if spots_status else 0
    cv2.putText(frame, f'Available spots: {available_spots} / {total_spots}', 
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

@app.on_event("startup")
async def startup_event():
    """Load model and mask on startup"""
    load_model_and_mask()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """Process uploaded video and return analysis results"""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return {"error": "Please upload a valid video file"}
    
    if PARKING_SPOTS is None:
        return {"error": "Parking detection system not properly initialized"}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}

        spots_status = [None for _ in PARKING_SPOTS]
        diffs = [None for _ in PARKING_SPOTS]
        previous_frame = None
        frame_nmr = 0
        step = 30
        processed_frames = []

        os.makedirs("static/output/frames", exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_nmr % step == 0 and previous_frame is not None:
                for spot_indx, spot in enumerate(PARKING_SPOTS):
                    x1, y1, w, h = spot
                    spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                    diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

            if frame_nmr % step == 0:
                if previous_frame is None:
                    arr_ = range(len(PARKING_SPOTS))
                else:
                    max_diff = np.amax(diffs) if diffs and any(d is not None for d in diffs) else 1
                    arr_ = [j for j in np.argsort(diffs) if diffs[j] and diffs[j] / max_diff > 0.4]

                for spot_indx in arr_:
                    x1, y1, w, h = PARKING_SPOTS[spot_indx]
                    spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                    spots_status[spot_indx] = empty_or_not(spot_crop)

            if frame_nmr % step == 0:
                previous_frame = frame.copy()

            if frame_nmr % 60 == 0:
                processed_frame = process_frame(frame.copy(), PARKING_SPOTS, spots_status)

                frame_filename = f"frame_{len(processed_frames) + 1}.jpg"
                frame_path = f"static/output/frames/{frame_filename}"
                cv2.imwrite(frame_path, processed_frame)

                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                processed_frames.append({
                    "base64": frame_base64,
                    "filename": frame_filename
                })

                if len(processed_frames) >= 10:
                    break

            frame_nmr += 1

        cap.release()

        available_spots = sum(1 for status in spots_status if status)
        total_spots = len(spots_status)
        occupancy_rate = ((total_spots - available_spots) / total_spots * 100) if total_spots else 0

        import pandas as pd
        output_data = {
            "Total Slots": [total_spots],
            "Occupied Slots": [total_spots - available_spots],
            "Available Slots": [available_spots],
            "Occupancy Rate (%)": [round(occupancy_rate, 1)]
        }
        os.makedirs("static/output", exist_ok=True)
        df = pd.DataFrame(output_data)
        df.to_csv("static/output/parking_analysis.csv", index=False)

        return {
            "success": True,
            "total_spots": total_spots,
            "available_spots": available_spots,
            "occupied_spots": total_spots - available_spots,
            "occupancy_rate": round(occupancy_rate, 1),
            "processed_frames": processed_frames,
            "frames_processed": frame_nmr
        }

    except Exception as e:
        return {"error": f"Error processing video: {str(e)}"}

    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)



# @app.get("/download-results")
# async def download_results():
#     csv_file_path = "static/output/parking_analysis.csv"
#     if os.path.exists(csv_file_path):
#         # os.makedirs("static/output/frames", exist_ok=True)
#         return FileResponse(csv_file_path, media_type='text/csv', filename="parking_analysis.csv")
#     return JSONResponse(content={"error": "Results not found. Please process a video first."}, status_code=404)

from fastapi.responses import FileResponse

# @app.get("/download-results")
# async def download_results():
#     csv_file_path = "static/output/parking_analysis.csv"
#     if os.path.exists(csv_file_path):
#         return FileResponse(csv_file_path, media_type='text/csv', filename="parking_analysis.csv")
#     return {"error": "CSV file not found. Please upload and process a video first."}


#gemini code
# application.py

# ... (rest of your imports and code) ...

@app.get("/download-results")
async def download_results():
    csv_file_path = "static/output/parking_analysis.csv"
    if os.path.exists(csv_file_path):
        return FileResponse(csv_file_path, media_type='text/csv', filename="parking_analysis.csv")
    return {"error": "CSV file not found. Please upload and process a video first."}

# ... (rest of your code) ...
# @app.get("/results", response_class=HTMLResponse)
# async def results_page(request: Request):
#     """Results page"""
#     return templates.TemplateResponse("results.html", {"request": request})
@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    csv_exists = os.path.exists("B:/Parking_Dec_Final/static/output/parking_analysis.csv")
    return templates.TemplateResponse("result.html", {
        "request": request,
        "csv_exists": csv_exists
    })


@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    return {
        "model_loaded": MODEL is not None,
        "mask_loaded": MASK is not None,
        "parking_spots_detected": len(PARKING_SPOTS) if PARKING_SPOTS else 0
    }
if __name__ == "__main__":
    import uvicorn
   # uvicorn.run(app, host="127.0.0.1", port=5000) - This is for Windows
    uvicorn.run(app, host="0.0.0.0", port=10000)

