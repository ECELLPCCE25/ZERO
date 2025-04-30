import logging
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, Query, HTTPException
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import asyncio
import threading
import queue
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime
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
supabase: Client = create_client(supabase_url, supabase_key)

router = APIRouter()
model = YOLO("yolov8n.pt")

# In-memory storage for connected cameras and person count data
connected_cameras: Dict[str, cv2.VideoCapture] = {}
analysis_results: Dict[str, Dict[str, Any]] = {}
results_lock = threading.Lock()
processing_queues: Dict[str, queue.Queue] = {}
processing_threads: Dict[str, threading.Thread] = {}
stop_event = threading.Event()

PERSON_COUNT_THRESHOLD = 5
SPATIAL_SMOOTHING_KERNEL_SIZE = (5, 5)
TEMPORAL_SMOOTHING_FRAMES = 2
QUIVER_STEP = 16
CONVERGENCE_THRESHOLD = -0.6
DIVERGENCE_THRESHOLD = 0.6
CURL_THRESHOLD = 0.2
MAGNITUDE_THRESHOLD = 0.8
OUTPUT_DIR = "analysis_output"
MAX_PROCESSING_DIMENSION = 240
NUM_GRID_CELLS = 15
DANGER_RECT_COLOR = (0, 80, 255)
DANGER_RECT_THICKNESS = 2
GRID_COLOR = (200, 200, 200)
GRID_THICKNESS = 1
GENERAL_DANGER_CELL_COLOR = (0, 0, 255)
GENERAL_DANGER_CELL_THICKNESS = 2
CONVERGENCE_THRESHOLD_TINT_COLOR = (100, 0, 0)
CONVERGENCE_THRESHOLD_OUTLINE_COLOR = (255, 255, 255)
CONVERGENCE_THRESHOLD_OUTLINE_THICKNESS = 1
PREDICTION_STEPS = 1
PREDICTION_COLOR = (0, 165, 255)
PREDICTION_RADIUS = 5
PREDICTION_FADE_FRAMES = 30
MORPH_KERNEL = np.ones((5, 5), np.uint8)
STREAM_DELAY_SECONDS = 0.1
IMAGE_TYPE_MAP = {
    "yolo": "yolo_output",
    "danger": "danger_zones",
    "motion": "motion_flow",
    "quiver": "quiver_plot",
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logging.info(f"Created output directory: {OUTPUT_DIR}")


def process_frames(stream_id: str, frame_queue: queue.Queue):
    logging.info(f"Processing thread started for stream ID: {stream_id}")
    prev_gray = None
    flow_history = deque(maxlen=TEMPORAL_SMOOTHING_FRAMES)
    active_predictions: List[Tuple[int, int, int]] = []
    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 6))

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
                if frame is None:
                    break

                original_height, original_width = frame.shape[:2]
                new_width, new_height = original_width, original_height

                if max(original_height, original_width) > MAX_PROCESSING_DIMENSION:
                    scale = MAX_PROCESSING_DIMENSION / max(
                        original_height, original_width
                    )
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    processed_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    processed_frame = frame.copy()

                yolo_results = model(processed_frame, verbose=False)
                person_count = sum(1 for r in yolo_results[0].boxes.cls if int(r) == 0)

                danger_regions = []
                convergence_points = []
                temporally_smoothed_flow = None
                divergence = None

                current_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None and prev_gray.shape == current_gray.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    flow_u_smooth = cv2.GaussianBlur(
                        flow[..., 0], SPATIAL_SMOOTHING_KERNEL_SIZE, 0
                    )
                    flow_v_smooth = cv2.GaussianBlur(
                        flow[..., 1], SPATIAL_SMOOTHING_KERNEL_SIZE, 0
                    )
                    spatially_smoothed_flow = np.stack(
                        (flow_u_smooth, flow_v_smooth), axis=-1
                    )
                    flow_history.append(spatially_smoothed_flow)
                    temporally_smoothed_flow = np.mean(list(flow_history), axis=0)

                    flow_u = temporally_smoothed_flow[..., 0]
                    flow_v = temporally_smoothed_flow[..., 1]

                    if flow_u.shape[0] > 1 and flow_u.shape[1] > 1:
                        du_dy, du_dx = np.gradient(flow_u)
                        dv_dy, dv_dx = np.gradient(flow_v)
                        divergence = du_dx + dv_dy
                        curl = dv_dx - du_dy
                        magnitude, angle = cv2.cartToPolar(
                            flow_u, flow_v, angleInDegrees=True
                        )

                        mask_divergence = divergence > DIVERGENCE_THRESHOLD
                        mask_curl = np.abs(curl) > CURL_THRESHOLD
                        mask_magnitude = magnitude > MAGNITUDE_THRESHOLD
                        mask_dangerous = (mask_divergence | mask_curl) & mask_magnitude

                        mask_dangerous_uint8 = mask_dangerous.astype(np.uint8) * 255
                        mask_dangerous_morphed = cv2.morphologyEx(
                            mask_dangerous_uint8, cv2.MORPH_OPEN, MORPH_KERNEL
                        )
                        mask_dangerous_morphed = cv2.morphologyEx(
                            mask_dangerous_morphed, cv2.MORPH_CLOSE, MORPH_KERNEL
                        )

                        contours, _ = cv2.findContours(
                            mask_dangerous_morphed,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            if w * h > 50:
                                danger_regions.append(
                                    (int(x), int(y), int(x + w), int(y + h))
                                )

                        convergence_mask = divergence < CONVERGENCE_THRESHOLD
                        convergence_points_coords = np.argwhere(convergence_mask)
                        convergence_points = [
                            (int(p[1]), int(p[0])) for p in convergence_points_coords
                        ]

                prev_gray = current_gray

                with results_lock:
                    if stream_id not in analysis_results:
                        analysis_results[stream_id] = {}
                    analysis_results[stream_id]["person_count"] = person_count
                    analysis_results[stream_id]["danger_regions"] = danger_regions
                    analysis_results[stream_id]["convergence_points"] = (
                        convergence_points
                    )
                    analysis_results[stream_id]["timestamp"] = (
                        datetime.now().isoformat()
                    )

                yolo_frame = processed_frame.copy()
                danger_frame = processed_frame.copy()

                if yolo_results and yolo_results[0].boxes:
                    for box in yolo_results[0].boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box[:4])
                        class_id = int(box[5])
                        confidence = box[4]
                        class_name = (
                            model.names[class_id]
                            if class_id in model.names
                            else str(class_id)
                        )
                        color = (0, 255, 0)
                        cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(
                            yolo_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                yolo_output_path = os.path.join(
                    OUTPUT_DIR, f"yolo_output_{stream_id}.png"
                )
                cv2.imwrite(yolo_output_path, yolo_frame)

                for x1, y1, x2, y2 in danger_regions:
                    cv2.rectangle(
                        danger_frame,
                        (x1, y1),
                        (x2, y2),
                        DANGER_RECT_COLOR,
                        DANGER_RECT_THICKNESS,
                    )

                current_height, current_width = processed_frame.shape[:2]
                aspect_ratio = current_width / current_height
                estimated_rows = round(math.sqrt(NUM_GRID_CELLS / aspect_ratio))
                estimated_cols = round(estimated_rows * aspect_ratio)
                grid_rows = max(1, estimated_rows)
                grid_cols = max(1, estimated_cols)
                cell_width = max(1, current_width // grid_cols)
                cell_height = max(1, current_height // grid_rows)

                is_cell_generally_dangerous = np.zeros(
                    (grid_rows, grid_cols), dtype=bool
                )
                current_convergence_cell_centers = []

                if divergence is not None:
                    for r in range(grid_rows):
                        for c in range(grid_cols):
                            cy1 = r * cell_height
                            cy2 = min(current_height, cy1 + cell_height)
                            cx1 = c * cell_width
                            cx2 = min(current_width, cx1 + cell_width)

                            for center_x, center_y in [
                                (x + w // 2, y + h // 2)
                                for (x, y, w, h) in danger_regions
                            ]:
                                if (
                                    original_width > MAX_PROCESSING_DIMENSION
                                    or original_height > MAX_PROCESSING_DIMENSION
                                ):
                                    scale_factor = current_width / original_width
                                    scaled_center_x = int(center_x * scale_factor)
                                    scaled_center_y = int(center_y * scale_factor)
                                else:
                                    scaled_center_x = center_x
                                    scaled_center_y = center_y

                                if (
                                    cx1 <= scaled_center_x < cx2
                                    and cy1 <= scaled_center_y < cy2
                                ):
                                    is_cell_generally_dangerous[r, c] = True
                                    break

                            if cy1 < cy2 and cx1 < cx2:
                                cell_divergence = divergence[cy1:cy2, cx1:cx2]
                                if (
                                    cell_divergence.size > 0
                                    and np.min(cell_divergence) < CONVERGENCE_THRESHOLD
                                ):
                                    current_convergence_cell_centers.append(
                                        (cx1 + cell_width // 2, cy1 + cell_height // 2)
                                    )

                new_predictions = []
                if temporally_smoothed_flow is not None:
                    for center_x, center_y in current_convergence_cell_centers:
                        int_center_y = int(np.clip(center_y, 0, current_height - 1))
                        int_center_x = int(np.clip(center_x, 0, current_width - 1))
                        flow_at_center_u = temporally_smoothed_flow[
                            int_center_y, int_center_x, 0
                        ]
                        flow_at_center_v = temporally_smoothed_flow[
                            int_center_y, int_center_x, 1
                        ]
                        predicted_center_x = (
                            center_x + flow_at_center_u * PREDICTION_STEPS
                        )
                        predicted_center_y = (
                            center_y + flow_at_center_v * PREDICTION_STEPS
                        )
                        predicted_center_x = np.clip(
                            predicted_center_x, 0, current_width - 1
                        )
                        predicted_center_y = np.clip(
                            predicted_center_y, 0, current_height - 1
                        )
                        new_predictions.append(
                            (
                                int(predicted_center_x),
                                int(predicted_center_y),
                                PREDICTION_FADE_FRAMES,
                            )
                        )

                active_predictions.extend(new_predictions)
                updated_predictions = []
                for pred_x, pred_y, age in active_predictions:
                    new_age = age - 1
                    if new_age > 0:
                        updated_predictions.append((pred_x, pred_y, new_age))
                active_predictions = updated_predictions

                for i in range(1, grid_cols):
                    x = i * cell_width
                    cv2.line(
                        danger_frame,
                        (x, 0),
                        (x, current_height),
                        GRID_COLOR,
                        GRID_THICKNESS,
                    )
                cv2.line(
                    danger_frame,
                    (current_width - 1, 0),
                    (current_width - 1, current_height),
                    GRID_COLOR,
                    GRID_THICKNESS,
                )

                for i in range(1, grid_rows):
                    y = i * cell_height
                    cv2.line(
                        danger_frame,
                        (0, y),
                        (current_width, y),
                        GRID_COLOR,
                        GRID_THICKNESS,
                    )
                cv2.line(
                    danger_frame,
                    (0, current_height - 1),
                    (current_width, current_height - 1),
                    GRID_COLOR,
                    GRID_THICKNESS,
                )

                for row in range(grid_rows):
                    for col in range(grid_cols):
                        if is_cell_generally_dangerous[row, col]:
                            cx1 = col * cell_width
                            cy1 = row * cell_height
                            cx2 = min(current_width, cx1 + cell_width)
                            cy2 = min(current_height, cy1 + cell_height)
                            cv2.rectangle(
                                danger_frame,
                                (cx1, cy1),
                                (cx2, cy2),
                                GENERAL_DANGER_CELL_COLOR,
                                GENERAL_DANGER_CELL_THICKNESS,
                            )

                if divergence is not None:
                    for r in range(grid_rows):
                        for c in range(grid_cols):
                            cy1 = r * cell_height
                            cy2 = min(current_height, cy1 + cell_height)
                            cx1 = c * cell_width
                            cx2 = min(current_width, cx1 + cell_width)
                            if cy1 < cy2 and cx1 < cx2:
                                cell_divergence = divergence[cy1:cy2, cx1:cx2]
                                if (
                                    cell_divergence.size > 0
                                    and np.min(cell_divergence) < CONVERGENCE_THRESHOLD
                                ):
                                    danger_frame[cy1:cy2, cx1:cx2] = np.clip(
                                        danger_frame[cy1:cy2, cx1:cx2].astype(
                                            np.float32
                                        )
                                        + CONVERGENCE_THRESHOLD_TINT_COLOR,
                                        0,
                                        255,
                                    ).astype(np.uint8)
                                    cv2.rectangle(
                                        danger_frame,
                                        (cx1, cy1),
                                        (cx2, cy2),
                                        CONVERGENCE_THRESHOLD_OUTLINE_COLOR,
                                        CONVERGENCE_THRESHOLD_OUTLINE_THICKNESS,
                                    )

                overlay = danger_frame.copy()
                for pred_x, pred_y, age in active_predictions:
                    alpha = age / PREDICTION_FADE_FRAMES
                    alpha = np.clip(alpha, 0, 1)
                    cv2.circle(
                        overlay,
                        (pred_x, pred_y),
                        PREDICTION_RADIUS,
                        PREDICTION_COLOR,
                        -1,
                    )
                    cv2.addWeighted(
                        overlay, alpha, danger_frame, 1 - alpha, 0, danger_frame
                    )

                danger_zones_path = os.path.join(
                    OUTPUT_DIR, f"danger_zones_{stream_id}.png"
                )
                cv2.imwrite(danger_zones_path, danger_frame)

                if temporally_smoothed_flow is not None:
                    magnitude, angle = cv2.cartToPolar(
                        temporally_smoothed_flow[..., 0],
                        temporally_smoothed_flow[..., 1],
                        angleInDegrees=True,
                    )
                    hsv = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                    hsv[..., 1] = 255
                    hsv[..., 0] = angle / 2
                    normalized_magnitude = cv2.normalize(
                        magnitude, None, 0, 255, cv2.NORM_MINMAX
                    )
                    hsv[..., 2] = normalized_magnitude
                    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    motion_flow_path = os.path.join(
                        OUTPUT_DIR, f"motion_flow_{stream_id}.png"
                    )
                    cv2.imwrite(motion_flow_path, rgb_flow)

                if temporally_smoothed_flow is not None:
                    h, w = new_height, new_width
                    y_coords, x_coords = np.mgrid[0:h:QUIVER_STEP, 0:w:QUIVER_STEP]
                    if x_coords.size > 0 and y_coords.size > 0:
                        u_vectors = temporally_smoothed_flow[y_coords, x_coords, 0]
                        v_vectors = temporally_smoothed_flow[y_coords, x_coords, 1]
                        valid_mask = np.isfinite(u_vectors) & np.isfinite(v_vectors)
                        x_coords_valid = x_coords[valid_mask]
                        y_coords_valid = y_coords[valid_mask]
                        u_vectors_valid = u_vectors[valid_mask]
                        v_vectors_valid = v_vectors[valid_mask]
                        ax.cla()
                        if x_coords_valid.size > 0:
                            angle_selected_rad = np.arctan2(
                                v_vectors_valid, u_vectors_valid
                            )
                            angle_selected_deg = (
                                np.degrees(angle_selected_rad) + 360
                            ) % 360
                            normalized_angles = angle_selected_deg / 360.0
                            cmap = cm.get_cmap("hsv")
                            colors = cmap(normalized_angles)
                            mag_selected = np.sqrt(
                                u_vectors_valid**2 + v_vectors_valid**2
                            )
                            max_mag_selected = (
                                np.max(mag_selected) if mag_selected.size > 0 else 1.0
                            )
                            quiver_scale = max_mag_selected / 50.0
                            ax.quiver(
                                x_coords_valid,
                                y_coords_valid,
                                u_vectors_valid,
                                v_vectors_valid,
                                color=colors,
                                scale=quiver_scale,
                                angles="xy",
                                scale_units="xy",
                                pivot="mid",
                            )
                            ax.set_title(f"Optical Flow Quiver Plot ({stream_id})")
                        else:
                            ax.set_title(
                                f"Optical Flow Quiver Plot ({stream_id}) - No Valid Vectors"
                            )
                        ax.invert_yaxis()
                        ax.set_aspect("equal", adjustable="box")
                        ax.set_xlim(0, new_width)
                        ax.set_ylim(new_height, 0)
                        ax.set_xlabel("X-coordinate")
                        ax.set_ylabel("Y-coordinate")
                        quiver_plot_path = os.path.join(
                            OUTPUT_DIR, f"quiver_plot_{stream_id}.png"
                        )
                        fig.savefig(quiver_plot_path)

            except queue.Empty:
                pass
            except Exception as e:
                logging.error(
                    f"Error in processing thread for stream ID {stream_id}: {e}"
                )
    finally:
        logging.info(f"Processing thread stopped for stream ID: {stream_id}")
        plt.close(fig)


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
                    latest_results = {
                        "person_count": 0,
                        "danger_regions": [],
                        "convergence_points": [],
                        "timestamp": datetime.now().isoformat(),
                        "message": "Waiting for analysis data...",
                    }
            await websocket.send_json(latest_results)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                logging.debug(f"Received message from WebSocket {stream_id}: {data}")
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
        logging.info(f"Streaming started for stream ID: {id}")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.warning(
                        f"Failed to read frame from stream ID: {id}. Ending stream."
                    )
                    break
                try:
                    processing_queues[id].put_nowait(frame.copy())
                except queue.Full:
                    logging.warning(
                        f"Processing queue for stream ID {id} is full. Dropping frame for processing."
                    )
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                await asyncio.sleep(STREAM_DELAY_SECONDS)
        except asyncio.CancelledError:
            logging.info(f"Streaming cancelled for stream ID: {id}")
        except Exception as e:
            logging.error(f"Error during streaming for stream ID {id}: {e}")
        finally:
            logging.info(f"Streaming ended for stream ID: {id}")
            if id in processing_queues:
                try:
                    processing_queues[id].put_nowait(None)
                except queue.Full:
                    pass
            if id in connected_cameras:
                connected_cameras[id].release()
                del connected_cameras[id]
            if id in processing_queues:
                with processing_queues[id].mutex:
                    processing_queues[id].queue.clear()

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@router.get("/analysis_stream/{stream_id}/{image_type}")
async def analysis_stream_endpoint(stream_id: str, image_type: str):
    if image_type not in IMAGE_TYPE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type: {image_type}. Valid types are: {list(IMAGE_TYPE_MAP.keys())}",
        )

    filename_prefix = IMAGE_TYPE_MAP[image_type]
    image_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_{stream_id}.png")

    async def generate_analysis_image():
        logging.info(
            f"Streaming analysis image '{image_type}' for stream ID: {stream_id}"
        )
        last_modified_time = 0
        try:
            while True:
                if os.path.exists(image_path):
                    current_modified_time = os.path.getmtime(image_path)
                    if current_modified_time > last_modified_time:
                        try:
                            with open(image_path, "rb") as f:
                                image_bytes = f.read()
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/png\r\n\r\n"
                                + image_bytes
                                + b"\r\n"
                            )
                            last_modified_time = current_modified_time
                        except Exception as e:
                            logging.error(
                                f"Error reading analysis image file {image_path}: {e}"
                            )
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logging.info(
                f"Analysis image streaming cancelled for stream ID: {stream_id}, type: {image_type}"
            )
        except Exception as e:
            logging.error(
                f"Error during analysis image streaming for stream ID {stream_id}, type {image_type}: {e}"
            )

    return StreamingResponse(
        generate_analysis_image(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@router.post("/stop_stream")
async def stop_stream(data: Dict[str, str]):
    stream_id = data.get("id")
    if not stream_id:
        raise HTTPException(
            status_code=400, detail="Missing stream 'id' in request body."
        )

    if stream_id in connected_cameras:
        logging.info(f"Stopping stream for ID: {stream_id}")
        connected_cameras[stream_id].release()
        if stream_id in processing_threads and processing_threads[stream_id].is_alive():
            logging.info(f"Waiting for processing thread {stream_id} to join...")
            processing_threads[stream_id].join(timeout=10.0)
            if processing_threads[stream_id].is_alive():
                logging.warning(
                    f"Processing thread {stream_id} did not stop gracefully."
                )
            else:
                logging.info(f"Processing thread {stream_id} joined successfully.")

        with results_lock:
            if stream_id in analysis_results:
                del analysis_results[stream_id]
                logging.info(f"Cleaned up analysis results for {stream_id}")

        if stream_id in processing_queues:
            with processing_queues[stream_id].mutex:
                processing_queues[stream_id].queue.clear()
            del processing_queues[stream_id]
            logging.info(f"Cleaned up processing queue for {stream_id}")

        if stream_id in processing_threads:
            del processing_threads[stream_id]
            logging.info(f"Cleaned up processing thread object for {stream_id}")

        return {
            "message": f"Stream and processing for ID {stream_id} stopped successfully"
        }
    else:
        raise HTTPException(
            status_code=404, detail=f"No active stream found for ID {stream_id}"
        )


def trigger_alert(stream_id, person_count):
    logging.warning(
        f"Alert! Person count ({person_count}) exceeds the threshold for stream ID: {stream_id}"
    )


def store_person_count_data(stream_id: str, user_id: str = None, name: str = None):
    logging.warning("store_person_count_data needs implementation.")
    pass


def cleanup():
    logging.info("Shutting down. Signaling processing threads to stop.")
    stop_event.set()
    for stream_id, thread in list(processing_threads.items()):
        if thread.is_alive():
            logging.info(f"Joining processing thread for {stream_id}")
            thread.join(timeout=10.0)
            if thread.is_alive():
                logging.warning(
                    f"Processing thread for {stream_id} did not stop gracefully."
                )
    for stream_id, cap in list(connected_cameras.items()):
        if cap.isOpened():
            logging.info(f"Releasing video capture for {stream_id}")
            cap.release()
            del connected_cameras[stream_id]
    logging.info("Cleanup complete.")


import atexit

atexit.register(cleanup)
