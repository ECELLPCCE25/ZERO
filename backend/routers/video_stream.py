import json
import logging
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, Query, HTTPException, Request
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Initialize the thread pool executor
executor = ThreadPoolExecutor(max_workers=10)

router = APIRouter()
model = YOLO("yolov8n.pt")

# In-memory storage for connected cameras and person count data
connected_cameras: Dict[str, cv2.VideoCapture] = {}
analysis_results: Dict[str, Dict[str, Any]] = {}
results_lock = threading.Lock()
processing_queues: Dict[str, queue.Queue] = {}
processing_threads: Dict[str, threading.Thread] = {}
stop_event = threading.Event()

# Threshold for triggering an alert
PERSON_COUNT_THRESHOLD = 2

sse_queue = asyncio.Queue()  # use this instead of a plain list

@router.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    logging.info(f"WebSocket connection accepted for stream ID: {stream_id}")
    try:
        while True:
            latest_results = {}
            with results_lock:
                if stream_id in analysis_results:
                    latest_results = analysis_results[stream_id]
                else:
                    # Process frame data if sent (optional)
                    frame = await asyncio.get_event_loop().run_in_executor(executor, decode_frame, data)
                    results = model(frame)
                    person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)

                    # Update person count data
                    if stream_id not in person_count_data:
                        person_count_data[stream_id] = []
                    person_count_data[stream_id].append((datetime.now(), person_count))

                    # Check if person count exceeds the threshold
                    if person_count > PERSON_COUNT_THRESHOLD:
                        await trigger_alert(stream_id, person_count, name="Camera")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logging.error(
                    f"Error receiving message from WebSocket {stream_id}: {e}"
                )
            await asyncio.sleep(0.1)
    except Exception as e:
        logging.error(
            f"WebSocket connection closed for stream ID: {stream_id} with error: {e}"
        )
    finally:
        logging.info(f"WebSocket connection closed for stream ID: {stream_id}")


@router.get("/ipcam")
async def ipcam_endpoint(
    device: str = Query(...),
    id: str = Query(...),
    user_id: str = Query(...),
    name: str = Query(...),
):
    if id in connected_cameras and connected_cameras[id].isOpened():
        logging.warning(f"Stream already active for ID: {id}")
        raise HTTPException(
            status_code=400, detail=f"Stream with ID {id} is already active."
        )

    # video_url = f"http://{device}:4747/video"
    video_url = f"dataset/2.mp4"
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        logging.error(f"Unable to open video stream from {video_url}")
        raise HTTPException(status_code=500, detail="Unable to open video stream")

    connected_cameras[id] = cap

    if id not in processing_queues:
        processing_queues[id] = queue.Queue(maxsize=10)

    if id not in processing_threads or not processing_threads[id].is_alive():
        logging.info(f"Starting processing thread for stream ID: {id}")
        processing_threads[id] = threading.Thread(
            target=process_frames, args=(id, processing_queues[id])
        )
        processing_threads[id].daemon = True
        processing_threads[id].start()

    async def generate():
        video_url = f"http://{device}:4747/video"
        cap = await asyncio.get_event_loop().run_in_executor(executor, cv2.VideoCapture, video_url)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Unable to open video stream")

        connected_cameras[device] = cap  # Store the video capture object

        try:
            while cap.isOpened():
                ret, frame = await asyncio.get_event_loop().run_in_executor(executor, cap.read)
                if not ret:
                    logging.warning(
                        f"Failed to read frame from stream ID: {id}. Ending stream."
                    )
                    break
                results = model(frame)
                person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)
                annotated_frame = await asyncio.get_event_loop().run_in_executor(executor, draw_boxes, frame, results)
                _, buffer = await asyncio.get_event_loop().run_in_executor(executor, cv2.imencode, '.jpg', annotated_frame)
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
                    await trigger_alert(id, person_count, name=name)
        finally:
            await asyncio.get_event_loop().run_in_executor(executor, cap.release)
            if device in connected_cameras:
                del connected_cameras[device]
            # Store data in Supabase when stream ends
            await store_person_count_data(id, user_id=user_id, name=name)
            # Clear person count data to prevent duplicates
            if id in person_count_data:
                del person_count_data[id]


@router.post("/stop_stream")
async def stop_stream(data: Dict[str, str]):
    ip = data.get("ip")
    if ip in connected_cameras:
        cap = connected_cameras[ip]
        await asyncio.get_event_loop().run_in_executor(executor, cap.release)
        del connected_cameras[ip]
        # Store data in Supabase when stream is stopped
        await store_person_count_data(ip)
        # Clear person count data to prevent duplicates
        if ip in person_count_data:
            del person_count_data[ip]
        return {"message": f"Stream for IP {ip} stopped successfully"}
    else:
        raise HTTPException(
            status_code=404, detail=f"No active stream found for ID {stream_id}"
        )

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

@router.get("/events")
async def stream_events(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break  # Stop if client disconnects

            data = await sse_queue.get()
            yield f"data: {json.dumps(data)}\n\n"
            sse_queue.task_done()
            yield data.message

    return StreamingResponse(event_generator(), media_type="text/event-stream")

async def alert_frontend(message: str):
    await sse_queue.put(message)
    print(f"Alerted Frontend: {message}")

async def trigger_alert(stream_id: str, person_count: int, name: str):
    """Triggers an alert by logging and sending a Supabase Realtime broadcast."""
    message = f"Persons overcrowded at {name}"
    logging.warning(f"ALERT! Stream ID: {stream_id}, Camera: {name}, Person Count: {person_count} (Threshold: {PERSON_COUNT_THRESHOLD})")
    alert_details = {
        "stream_id": stream_id,
        "person_count": person_count,
        "cam_name": name,
        "timestamp": datetime.now().isoformat(),
    }
    await alert_frontend(
        message=message,  # Send all data as a JSON string
    )

    try:
        # Use the async client and await the send operation
        await supabase.channel("camera_updates").send(
            {
                "type": "broadcast",
                "event": "UPDATE",
                "payload": {
                    "stream_id": stream_id,
                    "person_count": person_count,
                    "message": message,
                    "user_id": userId  # Ensure userId is passed correctly
                }
            }
        )
        logging.info(f"Realtime alert successfully sent for stream ID: {stream_id}")
    except Exception as e:
        logging.error(f"Failed to send Realtime alert for stream ID: {stream_id}: {e}")

async def store_person_count_data(stream_id: str, user_id: str = None, name: str = None):
    if stream_id in person_count_data and person_count_data[stream_id]:
        data_to_store = [{"timestamp": ts.isoformat(), "count": cnt} for ts, cnt in person_count_data[stream_id]]
        try:
            response = await asyncio.get_event_loop().run_in_executor(executor, supabase.table('cam_data').insert, [{
                'user_id': user_id,
                'stream_id': stream_id,
                'person_count_data': data_to_store,
                'cam_name': name
            }])
            logging.info(f"Data successfully stored in Supabase for stream ID: {stream_id}")
        except Exception as e:
            logging.error(f"Failed to store data in Supabase for stream ID: {stream_id}: {e}")
    else:
        logging.warning(f"No person count data found for stream ID: {stream_id}")


def get_current_person_count(stream_id: str) -> int:
    if stream_id in person_count_data and person_count_data[stream_id]:
        return person_count_data[stream_id][-1][1]
    return 0
