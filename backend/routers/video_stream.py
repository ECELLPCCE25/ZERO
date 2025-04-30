import logging
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, Query, HTTPException
from ultralytics import YOLO
import base64
from fastapi.responses import StreamingResponse
import asyncio
from typing import List, Dict
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)


router = APIRouter()
model = YOLO("yolov8n.pt")

# In-memory storage for connected cameras and person count data
connected_cameras = {}
person_count_data = {}

# Threshold for triggering an alert
PERSON_COUNT_THRESHOLD = 2

@router.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    try:
        while True:
            # Send person count data every second
            response_data = {
                "person_count_data": get_last_15_seconds_data(person_count_data.get(stream_id, [])),
                "current_person_count": get_current_person_count(stream_id),
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(response_data)

            # Check for incoming messages (e.g., "request_person_count" or frame data)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                if data == "request_person_count":
                    # Already sent above, no need to resend
                    continue
                else:
                    # Process frame data if sent (optional)
                    frame = decode_frame(data)
                    results = model(frame)
                    person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)

                    # Update person count data
                    if stream_id not in person_count_data:
                        person_count_data[stream_id] = []
                    person_count_data[stream_id].append((datetime.now(), person_count))

                    # Check if person count exceeds the threshold
                    if person_count > PERSON_COUNT_THRESHOLD:
                        trigger_alert(stream_id, person_count)
            except asyncio.TimeoutError:
                # No message received, continue sending periodic updates
                pass

            await asyncio.sleep(1)  # Send data every second
    except Exception as e:
        logging.error(f"WebSocket connection closed for stream ID: {stream_id} with error: {e}")
        # Optionally store data here if WebSocket is the only connection
        # store_person_count_data(stream_id)
    finally:
        # Optionally store data here, but primary storage should be in /ipcam
        # store_person_count_data(stream_id)
        pass

@router.get("/ipcam")
async def ipcam_endpoint(device: str = Query(...), id: str = Query(...), user_id: str = Query(...), name: str = Query(...)):
    async def generate():
        video_url = f"http://{device}:4747/video"
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Unable to open video stream")

        connected_cameras[device] = cap  # Store the video capture object

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
                    trigger_alert(id, person_count, name=name)
        finally:
            cap.release()
            if device in connected_cameras:
                del connected_cameras[device]
            # Store data in Supabase when stream ends
            store_person_count_data(id, user_id=user_id, name=name)
            # Clear person count data to prevent duplicates
            if id in person_count_data:
                del person_count_data[id]

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

@router.post("/stop_stream")
async def stop_stream(data: Dict[str, str]):
    ip = data.get("ip")
    if ip in connected_cameras:
        cap = connected_cameras[ip]
        cap.release()
        del connected_cameras[ip]
        # Store data in Supabase when stream is stopped
        store_person_count_data(ip)
        # Clear person count data to prevent duplicates
        if ip in person_count_data:
            del person_count_data[ip]
        return {"message": f"Stream for IP {ip} stopped successfully"}
    else:
        raise HTTPException(status_code=404, detail="No active stream found for the specified IP")


def decode_frame(data):
    encoded_data = data.split(",")[1]
    np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

def draw_boxes(frame, results):
    annotated_frame = frame.copy()
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated_frame

def trigger_alert(stream_id, person_count, name: str):
    logging.warning(f"Alert! Person count ({person_count}) exceeds the threshold for stream ID: {stream_id}")

    # Use async method to broadcast to Supabase Realtime
    supabase.channel("camera_updates").send(
        {
            "type": "broadcast",
            "event": "UPDATE",
            "payload": {
                "stream_id": stream_id,
                "person_count": person_count,
                "message": f"Persons overcrowded at {name}",
            }
        }
    )


def store_person_count_data(stream_id: str, user_id: str = None, name: str = None):
    if stream_id in person_count_data and person_count_data[stream_id]:
        data_to_store = [{"timestamp": ts.isoformat(), "count": cnt} for ts, cnt in person_count_data[stream_id]]
        try:
            response = supabase.table('cam_data').insert([{
                'user_id': user_id,
                'stream_id': stream_id,
                'person_count_data': data_to_store,
                'cam_name': name
            }]).execute()
            logging.info(f"Data successfully stored in Supabase for stream ID: {stream_id}")
        except Exception as e:
            logging.error(f"Failed to store data in Supabase for stream ID: {stream_id}: {e}")
    else:
        logging.warning(f"No person count data found for stream ID: {stream_id}")

def get_last_15_seconds_data(data):
    now = datetime.now()
    fifteen_seconds_ago = now - timedelta(seconds=15)
    return [{"timestamp": ts.isoformat(), "count": cnt} for ts, cnt in data if ts > fifteen_seconds_ago]

def get_current_person_count(stream_id: str) -> int:
    if stream_id in person_count_data and person_count_data[stream_id]:
        return person_count_data[stream_id][-1][1]
    return 0
