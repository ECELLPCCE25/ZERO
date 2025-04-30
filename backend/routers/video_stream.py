import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, Query, HTTPException
from ultralytics import YOLO
import base64
from fastapi.responses import StreamingResponse
import asyncio
from typing import List, Dict

router = APIRouter()
model = YOLO("yolov8n.pt")

# In-memory storage for connected cameras
connected_cameras = {}


@router.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        frame = decode_frame(data)
        results = model(frame)
        person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)
        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            f"People Count: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        encoded_frame = encode_frame(annotated_frame)
        await websocket.send_text(encoded_frame)


@router.get("/ipcam")
async def ipcam_endpoint(device: str = Query(...), id: str = Query(...)):
    async def generate():
        video_url = f"http://{device}:4747/video"
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Unable to open video stream")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)
            annotated_frame = results[0].plot()
            print(f"People_count: {person_count}")
            cv2.putText(
                annotated_frame,
                f"People Count: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            await asyncio.sleep(0.03)
        cap.release()

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


def decode_frame(data):
    encoded_data = data.split(",")[1]
    np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    encoded_frame = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_frame}"
