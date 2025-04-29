import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, Query, HTTPException
from ultralytics import YOLO
import base64
from fastapi.responses import StreamingResponse
import asyncio
from typing import List, Dict
from datetime import datetime

router = APIRouter()
model = YOLO("yolov8n.pt")

# In-memory storage for connected cameras and person count data
connected_cameras = {}
person_count_data = {}

# Threshold for triggering an alert
PERSON_COUNT_THRESHOLD = 5

@router.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame = decode_frame(data)
            results = model(frame)
            person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)
            annotated_frame = draw_boxes(frame, results)
            encoded_frame = encode_frame(annotated_frame)
            await websocket.send_text(encoded_frame)

            # Track person count data
            if stream_id not in person_count_data:
                person_count_data[stream_id] = []
            person_count_data[stream_id].append((datetime.now(), person_count))

            # Check if person count exceeds the threshold
            if person_count > PERSON_COUNT_THRESHOLD:
                trigger_alert(stream_id, person_count)
    except Exception as e:
        print(f"WebSocket connection closed for stream ID: {stream_id}")
        print_person_count_data(stream_id)

@router.get("/ipcam")
async def ipcam_endpoint(device: str = Query(...), id: str = Query(...)):
    async def generate():
        video_url = f"http://{device}:4747/video"
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Unable to open video stream")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)
                annotated_frame = draw_boxes(frame, results)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                await asyncio.sleep(0.03)

                # Track person count data
                if id not in person_count_data:
                    person_count_data[id] = []
                person_count_data[id].append((datetime.now(), person_count))

                # Check if person count exceeds the threshold
                if person_count > PERSON_COUNT_THRESHOLD:
                    trigger_alert(id, person_count)
        finally:
            cap.release()
            print_person_count_data(id)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

@router.get("/person_count_data/{stream_id}")
async def get_person_count_data(stream_id: str):
    if stream_id in person_count_data:
        return {"data": person_count_data[stream_id]}
    else:
        raise HTTPException(status_code=404, detail="No data found for the specified stream ID")

def decode_frame(data):
    encoded_data = data.split(",")[1]
    np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_frame}"

def draw_boxes(frame, results):
    annotated_frame = frame.copy()
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated_frame

def trigger_alert(stream_id, person_count):
    print(f"Alert! Person count ({person_count}) exceeds the threshold for stream ID: {stream_id}")
    # Add additional alert logic here (e.g., send a notification, log the event, etc.)

def print_person_count_data(stream_id):
    if stream_id in person_count_data:
        print(f"Person count data for stream ID {stream_id}:")
        for timestamp, count in person_count_data[stream_id]:
            print(f"{timestamp}: {count}")
    else:
        print(f"No person count data found for stream ID {stream_id}")
