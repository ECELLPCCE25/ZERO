from lib.motion import (
    MAX_WIDTH,
    NUM_GRID_CELLS,
    TEMPORAL_SMOOTHING_FRAMES,
    process_frame,
)
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
from fastapi import APIRouter, WebSocket, Query, HTTPException

from fastapi.responses import StreamingResponse


router = APIRouter()


def initialize_video_capture(source="dataset/5.mp4"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()
    return cap


def calculate_frame_dimensions(cap, max_width=MAX_WIDTH):
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if original_width > max_width:
        new_width = max_width
        new_height = int(original_height * (max_width / original_width))
    else:
        new_width = original_width
        new_height = original_height

    print(f"Original dimensions: {original_width}x{original_height}")
    print(f"Processing dimensions: {new_width}x{new_height}")
    return new_width, new_height


def read_first_frame(cap, new_width, new_height):
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        exit()
    frame1_resized = cv2.resize(frame1, (new_width, new_height))
    prev_gray = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    return prev_gray


def initialize_hsv_image(new_width, new_height):
    hsv = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    return hsv


def calculate_grid_dimensions(new_width, new_height):
    aspect_ratio = new_width / new_height
    estimated_rows = round(math.sqrt(NUM_GRID_CELLS / aspect_ratio))
    estimated_cols = round(estimated_rows * aspect_ratio)

    grid_rows = max(1, estimated_rows)
    grid_cols = max(1, estimated_cols)

    if grid_rows * grid_cols < NUM_GRID_CELLS:
        if aspect_ratio > 1:
            grid_cols = max(1, grid_cols + 1)
        else:
            grid_rows = max(1, grid_rows + 1)

    cell_width = max(1, new_width // grid_cols)
    cell_height = max(1, new_height // grid_rows)

    print(
        f"Calculated grid size: {grid_rows}x{grid_cols} ({grid_rows * grid_cols} cells)"
    )
    print(f"Cell dimensions: {cell_width}x{cell_height}")
    return grid_rows, grid_cols, cell_width, cell_height


def setup_matplotlib():
    plt.figure("Optical Flow Quiver Plot")
    plt.ion()
    return plt.gca()


@router.get("/motioncam")
def motion_flow(
    device: str = Query(...),
    id: str = Query(...),
    user_id: str = Query(...),
    name: str = Query(...),
):
    def generate():
        video_url = f"http://{device}:4747/video"
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Unable to open video stream")

        new_width, new_height = calculate_frame_dimensions(cap)
        prev_gray = read_first_frame(cap, new_width, new_height)
        hsv = initialize_hsv_image(new_width, new_height)
        flow_history = deque(maxlen=TEMPORAL_SMOOTHING_FRAMES)
        active_predictions = []
        grid_rows, grid_cols, cell_width, cell_height = calculate_grid_dimensions(
            new_width, new_height
        )

        while True:
            ret, prev_gray, output_frame, quiver = process_frame(
                cap,
                prev_gray,
                new_width,
                new_height,
                hsv,
                flow_history,
                active_predictions,
                grid_rows,
                grid_cols,
                cell_width,
                cell_height,
                plt.gca(),  # No matplotlib axis
            )
            if not ret:
                break

            _, buffer = cv2.imencode(".jpg", output_frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        cap.release()

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@router.get("/quivercam")
def motion_flow(
    device: str = Query(...),
    id: str = Query(...),
    user_id: str = Query(...),
    name: str = Query(...),
):
    def generate():
        video_url = f"http://{device}:4747/video"
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Unable to open video stream")

        new_width, new_height = calculate_frame_dimensions(cap)
        prev_gray = read_first_frame(cap, new_width, new_height)
        hsv = initialize_hsv_image(new_width, new_height)
        flow_history = deque(maxlen=TEMPORAL_SMOOTHING_FRAMES)
        active_predictions = []
        grid_rows, grid_cols, cell_width, cell_height = calculate_grid_dimensions(
            new_width, new_height
        )

        while True:
            ret, prev_gray, output_frame, quiver = process_frame(
                cap,
                prev_gray,
                new_width,
                new_height,
                hsv,
                flow_history,
                active_predictions,
                grid_rows,
                grid_cols,
                cell_width,
                cell_height,
                plt.gca(),  # No matplotlib axis
            )
            if not ret:
                break

            _, buffer = cv2.imencode(".jpg", quiver)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        cap.release()

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace;boundary=frame"
    )
