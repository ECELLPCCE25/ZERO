import json
import logging
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, Query, HTTPException, FastAPI
from ultralytics import YOLO
import base64
from fastapi.responses import StreamingResponse
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
# Use create_client from the main library for sync operations
from supabase import create_client as create_sync_client, Client
from dotenv import load_dotenv
import os
import uvicorn # For running the app

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Supabase clients
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logging.error("Supabase URL or Key not found in environment variables. Please check your .env file.")
    exit(1)

supabase: Client = create_sync_client(supabase_url, supabase_key) # Sync client for DB writes

# --- FastAPI Setup ---
router = APIRouter()

# --- YOLO Model ---
try:
    model = YOLO("yolov8n.pt") # Or your custom model path
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    exit(1)


# --- In-memory Storage ---
# Stores {device_ip: (capture_object, stream_id, user_id, cam_name)}
connected_cameras: Dict[str, Tuple[cv2.VideoCapture, str, str, str]] = {}
# Stores {stream_id: [(timestamp, count)]}
person_count_data: Dict[str, List[Tuple[datetime, int]]] = {}

# --- Constants ---
PERSON_COUNT_THRESHOLD = 2 # Trigger alert if person count exceeds this

# --- Helper Functions ---

def decode_frame(data: str) -> Optional[np.ndarray]:
    """Decodes a base64 encoded frame string."""
    try:
        encoded_data = data.split(",")[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logging.error(f"Error decoding frame: {e}")
        return None

def draw_boxes(frame: np.ndarray, results) -> np.ndarray:
    """Draws bounding boxes on the frame based on YOLO results."""
    annotated_frame = frame.copy()
    # Accessing boxes according to ultralytics v8 results structure
    person_boxes = [box for box in results[0].boxes if int(box.cls) == 0] # Class 0 is 'person' in COCO
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        label = f"{model.names[int(box.cls)]} {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_frame

sse_queue = asyncio.Queue()  # use this instead of a plain list

@router.get("/events")
async def stream_events(request):
    """
    Endpoint to stream server-sent events.
    Keeps connection open and sends data from sse_queue.
    """
    async def event_generator():
        """Yields events from the queue as they become available."""
        while True:
            try:
                if await request.is_disconnected():
                    logging.info("SSE client disconnected.")
                    break

                event_data = await sse_queue.get()
                logging.info(f"Sending SSE data to client: {event_data}")
                yield f"data: {json.dumps(event_data)}\n\n"
                sse_queue.task_done()

            except asyncio.CancelledError:
                logging.info("SSE event generator cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in SSE event generator: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")



async def alert_frontend(message: str):
    await sse_queue.put({"message": message})

    # sse_clients.append({"message": message, "progress": progress,"type":type})
    print(f"Alerted Frontend: {message} (%)")
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
    alert_frontend(
        message=json.dumps(alert_details), # Send all data as a JSON string
    )

    try:
        # Use the async client and await the send operation
      
        logging.info(f"Realtime alert successfully sent for stream ID: {stream_id}")
    except Exception as e:
        logging.error(f"Failed to send Realtime alert for stream ID: {stream_id}: {e}")

def store_person_count_data(stream_id: str, user_id: Optional[str] = None, name: Optional[str] = None):
    """Stores accumulated person count data to Supabase using the sync client."""
    if stream_id in person_count_data and person_count_data[stream_id]:
        # Prepare data in a JSON-compatible format
        data_to_store = [{"timestamp": ts.isoformat(), "count": cnt} for ts, cnt in person_count_data[stream_id]]
        try:
            insert_payload = {
                'stream_id': stream_id,
                'person_count_data': data_to_store,
            }
            if user_id:
                insert_payload['user_id'] = user_id
            if name:
                insert_payload['cam_name'] = name

            # Use the synchronous client for insertion
            response = supabase.table('cam_data').insert([insert_payload]).execute()
            logging.info(f"Data successfully stored in Supabase for stream ID: {stream_id}. Response: {response}")
            # Clear the stored data from memory after successful DB insertion
            del person_count_data[stream_id]
        except Exception as e:
            logging.error(f"Failed to store data in Supabase for stream ID: {stream_id}: {e}")
    else:
        logging.warning(f"No person count data found in memory for stream ID: {stream_id} to store.")

def get_last_15_seconds_data(stream_id: str) -> List[Dict]:
    """Retrieves person count data for the last 15 seconds for a given stream."""
    if stream_id not in person_count_data:
        return []
    now = datetime.now()
    fifteen_seconds_ago = now - timedelta(seconds=15)
    data = person_count_data[stream_id]
    # Filter data and format it
    return [{"timestamp": ts.isoformat(), "count": cnt} for ts, cnt in data if ts > fifteen_seconds_ago]

def get_current_person_count(stream_id: str) -> int:
    """Gets the most recent person count for a given stream."""
    if stream_id in person_count_data and person_count_data[stream_id]:
        return person_count_data[stream_id][-1][1] # Get count from the last recorded tuple
    return 0

# --- FastAPI Lifecycle Events ---
@router.on_event("startup")
async def startup_event():
    logging.info("Application startup: Initializing resources.")
    # Potentially connect async client if needed (supabase-py v2 often handles connections implicitly)
    pass

@router.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutdown: Cleaning up resources.")
    # Gracefully close the async client's resources
    try:
        # await supabase_async.postgrest.aclose() # Close the underlying httpx client
        logging.info("Async Supabase client resources closed.")
    except Exception as e:
        logging.error(f"Error closing async Supabase client: {e}")
    # Release any remaining camera captures
    for device, (cap, stream_id, _, _) in list(connected_cameras.items()): # Iterate over a copy
        logging.info(f"Releasing camera capture for device {device} (Stream ID: {stream_id}) on shutdown.")
        cap.release()
        del connected_cameras[device]


# --- API Endpoints ---

@router.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint to stream real-time person count data."""
    await websocket.accept()
    logging.info(f"WebSocket connection established for stream ID: {stream_id}")
    try:
        while True:
            # Check for incoming messages (optional, e.g., client sending frame data)
            try:
                # Set a timeout to avoid blocking indefinitely if client sends nothing
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                logging.debug(f"Received message from WebSocket {stream_id}: {message[:50]}...") # Log snippet
                # --- Optional: Process incoming frame data from WebSocket ---
                # if message != "request_data": # Example condition
                #     frame = decode_frame(message)
                #     if frame is not None:
                #         results = model(frame)
                #         person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)
                #         now = datetime.now()
                #         if stream_id not in person_count_data:
                #             person_count_data[stream_id] = []
                #         person_count_data[stream_id].append((now, person_count))
                #         # Trim old data (e.g., keep only last minute)
                #         one_minute_ago = now - timedelta(minutes=1)
                #         person_count_data[stream_id] = [(ts, cnt) for ts, cnt in person_count_data[stream_id] if ts > one_minute_ago]
                #         # Optional: Trigger alert based on WebSocket frame analysis
                #         # if person_count > PERSON_COUNT_THRESHOLD:
                #         #     await trigger_alert(stream_id, person_count, "WebSocket Camera") # Need name logic
            except asyncio.TimeoutError:
                # No message received from client, continue loop
                pass
            except Exception as e:
                # Handle other WebSocket errors (like disconnect)
                logging.error(f"WebSocket error for stream ID {stream_id}: {e}")
                break # Exit loop on error

            # Send current data
            response_data = {
                "person_count_data_last_15s": get_last_15_seconds_data(stream_id),
                "current_person_count": get_current_person_count(stream_id),
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(response_data)
            await asyncio.sleep(1) # Send updates every second

    except Exception as e:
        logging.error(f"WebSocket connection closed for stream ID: {stream_id}. Error: {e}")
    finally:
        logging.info(f"WebSocket connection terminated for stream ID: {stream_id}")
        # Note: Data storage primarily happens when the /ipcam stream ends or is stopped.
        # You might add logic here if WS is the *only* source for a stream_id.

@router.get("/ipcam")
async def ipcam_endpoint(device: str = Query(..., description="IP address and port of the camera (e.g., 192.168.1.10:4747)"),
                         id: str = Query(..., description="Unique identifier for this camera stream"),
                         user_id: str = Query(..., description="ID of the user associated with this camera"),
                         name: str = Query(..., description="User-friendly name for the camera")):
    """Connects to an IP camera, performs person detection, and streams annotated video."""
    video_url = f"http://{device}/video" # Assuming DroidCam URL structure
    logging.info(f"Attempting to connect to video stream: {video_url} for Stream ID: {id}")

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logging.error(f"Unable to open video stream: {video_url}")
        raise HTTPException(status_code=503, detail=f"Unable to open video stream at {video_url}")

    if device in connected_cameras:
        logging.warning(f"Device {device} already has an active stream. Closing existing one.")
        old_cap, old_id, old_user, old_name = connected_cameras[device]
        old_cap.release()
        # Store data for the stream being replaced
        store_person_count_data(old_id, old_user, old_name)
        del connected_cameras[device]


    # Store capture object and metadata associated with the device IP
    connected_cameras[device] = (cap, id, user_id, name)
    logging.info(f"Stream started successfully for Device: {device}, Stream ID: {id}, Camera Name: {name}")


    async def generate():
        stream_id = id # Use the provided id consistently
        stream_user_id = user_id
        stream_name = name
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Failed to retrieve frame from {video_url}. Stream ending.")
                    break

                # Perform detection
                results = model(frame, verbose=False) # verbose=False reduces console spam
                person_count = sum(1 for r in results[0].boxes if int(r.cls) == 0) # Class 0 is 'person'

                # Annotate frame
                annotated_frame = draw_boxes(frame, results)
                # Add person count text to the frame
                cv2.putText(annotated_frame, f"Persons: {person_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    logging.warning("Failed to encode frame to JPEG.")
                    continue
                frame_bytes = buffer.tobytes()

                # Yield the frame for the streaming response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Track person count data using the unique stream_id
                now = datetime.now()
                if stream_id not in person_count_data:
                    person_count_data[stream_id] = []
                person_count_data[stream_id].append((now, person_count))

                # Simple data trimming (e.g., keep last 5 minutes of data points in memory)
                five_minutes_ago = now - timedelta(minutes=5)
                person_count_data[stream_id] = [(ts, cnt) for ts, cnt in person_count_data[stream_id] if ts > five_minutes_ago]


                # Check if person count exceeds the threshold
                if person_count > PERSON_COUNT_THRESHOLD:
                    # Call the async trigger_alert function
                    await trigger_alert(stream_id, person_count, stream_name)

                # Control frame rate - adjust sleep time as needed
                await asyncio.sleep(0.05) # ~20 FPS target, adjust based on processing time

        except Exception as e:
            logging.error(f"Error during video generation for stream {stream_id} ({video_url}): {e}")
        finally:
            logging.info(f"Closing video stream and cleaning up for device {device} (Stream ID: {stream_id}).")
            cap.release()
            if device in connected_cameras:
                # Ensure we're removing the correct entry
                _, stored_id, _, _ = connected_cameras[device]
                if stored_id == stream_id:
                    del connected_cameras[device]
                else:
                    logging.warning(f"Device {device} mapping changed unexpectedly during stream.")

            # Store accumulated data in Supabase when stream ends
            logging.info(f"Attempting to store data for stream ID: {stream_id}")
            store_person_count_data(stream_id, stream_user_id, stream_name)
            # Note: person_count_data[stream_id] is deleted inside store_person_count_data on success


    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")


@router.post("/stop_stream", status_code=200)
async def stop_stream(data: Dict[str, str]):
    """Stops a specific camera stream identified by its device IP address."""
    device = data.get("device") # Expect 'device' (IP:Port) in the request body
    if not device:
        raise HTTPException(status_code=400, detail="Missing 'device' (IP:Port) in request body")

    if device in connected_cameras:
        cap, stream_id, user_id, name = connected_cameras[device]
        logging.info(f"Stopping stream requested for device {device} (Stream ID: {stream_id}).")
        cap.release()
        del connected_cameras[device]

        # Store remaining data for this stream
        logging.info(f"Storing final data for stream ID: {stream_id} due to stop request.")
        store_person_count_data(stream_id, user_id, name)
        # Note: person_count_data[stream_id] is deleted inside store_person_count_data on success

        return {"message": f"Stream for device {device} (ID: {stream_id}) stopped successfully and data stored."}
    else:
        logging.warning(f"Stop stream request: No active stream found for device {device}.")
        raise HTTPException(status_code=404, detail=f"No active stream found for the specified device IP: {device}")


