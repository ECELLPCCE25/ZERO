import logging
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO
import base64
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import math

router = APIRouter()

try:
    model = YOLO("yolov8n.pt")
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    model = None

SMOOTHING_KERNEL = (5, 5)

CONVERGENCE_THRESH = -0.6
DIVERGENCE_THRESH = 0.6
CURL_THRESH = 0.2
MAGNITUDE_THRESH = 0.8
DANGER_BOX_COLOR = (0, 80, 255)
DANGER_BOX_THICKNESS = 2

GRID_CELL_COUNT = 15
GRID_LINE_COLOR = (200, 200, 200)
GRID_LINE_THICKNESS = 1

DANGER_CELL_COLOR = (0, 0, 255)
DANGER_CELL_THICKNESS = 2

CONVERGENCE_TINT_COLOR = (100, 0, 0)
CONVERGENCE_OUTLINE_COLOR = (255, 255, 255)
CONVERGENCE_OUTLINE_THICKNESS = 1

MAX_DIMENSION = 640
MORPH_STRUCT_ELEMENT = np.ones((5, 5), np.uint8)

gpu_available = False
try:
    cv2.cuda.GpuMat()
    logging.info("OpenCV CUDA device detected and initialized.")
    gpu_available = True
except cv2.error as e:
    logging.warning(
        f"OpenCV CUDA not available or initialized: {e}. Falling back to CPU."
    )
    gpu_available = False
except Exception as e:
    logging.error(
        f"An unexpected error occurred checking for CUDA: {e}. Falling back to CPU."
    )
    gpu_available = False


def process_gpu(
    prev_frame_gpu: cv2.cuda_GpuMat,
    current_frame_gpu: cv2.cuda_GpuMat,
    proc_width: int,
    proc_height: int,
    grid_rows: int,
    grid_cols: int,
    cell_width: int,
    cell_height: int,
    detection_results: Any,
) -> tuple[np.ndarray, Dict[str, Any]]:
    height, width = proc_height, proc_width

    current_gray_gpu = cv2.cuda.cvtColor(current_frame_gpu, cv2.COLOR_BGR2GRAY)

    flow_calc = cv2.cuda_FarnebackOpticalFlow.create()
    flow_data_gpu = flow_calc.calc(prev_frame_gpu, current_gray_gpu, None)

    stream = cv2.cuda_Stream()
    blur_filter_gpu = cv2.cuda.createGaussianFilter(
        cv2.CV_32FC2, cv2.CV_32FC2, SMOOTHING_KERNEL, 0
    )
    smoothed_flow_gpu = blur_filter_gpu.apply(flow_data_gpu, stream=stream)

    smoothed_flow_cpu = smoothed_flow_gpu.download(stream=stream)
    stream.waitForCompletion()

    flow_x = smoothed_flow_cpu[..., 0]
    flow_y = smoothed_flow_cpu[..., 1]

    if height < 2 or width < 2:
        divergence = np.zeros_like(flow_x)
        curl = np.zeros_like(flow_y)
    else:
        dy_x, dx_x = np.gradient(flow_x)
        dy_y, dx_y = np.gradient(flow_y)
        divergence = dx_x + dy_y
        curl = dx_y - dy_x

    magnitude = np.sqrt(flow_x**2 + flow_y**2)

    mask_div = divergence > DIVERGENCE_THRESH
    mask_crl = np.abs(curl) > CURL_THRESH
    mask_mag = magnitude > MAGNITUDE_THRESH
    mask_critical = (mask_div | mask_crl) & mask_mag
    mask_critical_uint8 = mask_critical.astype(np.uint8) * 255
    mask_critical_morphed = cv2.morphologyEx(
        mask_critical_uint8, cv2.MORPH_OPEN, MORPH_STRUCT_ELEMENT
    )
    mask_critical_morphed = cv2.morphologyEx(
        mask_critical_morphed, cv2.MORPH_CLOSE, MORPH_STRUCT_ELEMENT
    )

    contours, _ = cv2.findContours(
        mask_critical_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    danger_boxes = []
    critical_centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 50:
            danger_boxes.append({"x": x, "y": y, "w": w, "h": h})
            critical_centers.append((x + w // 2, y + h // 2))

    critical_cells = []
    converging_cells = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * cell_height
            y2 = min(height, y1 + cell_height)
            x1 = c * cell_width
            x2 = min(width, x1 + cell_width)

            if y1 < y2 and x1 < x2:
                for center_x, center_y in critical_centers:
                    if x1 <= center_x < x2 and y1 <= center_y < y2:
                        critical_cells.append({"row": r, "col": c})
                        break

                cell_divergence = divergence[y1:y2, x1:x2]
                if (
                    cell_divergence.size > 0
                    and np.min(cell_divergence) < CONVERGENCE_THRESH
                ):
                    converging_cells.append({"row": r, "col": c})

    current_frame_cpu = current_frame_gpu.download(stream=stream)
    stream.waitForCompletion()

    annotated_frame = current_frame_cpu.copy()

    annotated_frame = draw_boxes(annotated_frame, detection_results)

    for box in danger_boxes:
        cv2.rectangle(
            annotated_frame,
            (box["x"], box["y"]),
            (box["x"] + box["w"], box["y"] + box["h"]),
            DANGER_BOX_COLOR,
            DANGER_BOX_THICKNESS,
        )

    for i in range(1, grid_cols):
        x = i * cell_width
        cv2.line(
            annotated_frame, (x, 0), (x, height), GRID_LINE_COLOR, GRID_LINE_THICKNESS
        )
    cv2.line(
        annotated_frame,
        (width - 1, 0),
        (width - 1, height),
        GRID_LINE_COLOR,
        GRID_LINE_THICKNESS,
    )

    for i in range(1, grid_rows):
        y = i * cell_height
        cv2.line(
            annotated_frame, (0, y), (width, y), GRID_LINE_COLOR, GRID_LINE_THICKNESS
        )
    cv2.line(
        annotated_frame,
        (0, height - 1),
        (width, height - 1),
        GRID_LINE_COLOR,
        GRID_LINE_THICKNESS,
    )

    for cell_idx in critical_cells:
        r, c = cell_idx["row"], cell_idx["col"]
        x1 = c * cell_width
        y1 = r * cell_height
        x2 = min(width, x1 + cell_width)
        y2 = min(height, y1 + cell_height)
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            DANGER_CELL_COLOR,
            DANGER_CELL_THICKNESS,
        )

    for cell_idx in converging_cells:
        r, c = cell_idx["row"], cell_idx["col"]
        y1 = r * cell_height
        y2 = min(height, y1 + cell_height)
        x1 = c * cell_width
        x2 = min(width, x1 + cell_width)
        annotated_frame[y1:y2, x1:x2] = np.clip(
            annotated_frame[y1:y2, x1:x2].astype(np.float32) + CONVERGENCE_TINT_COLOR,
            0,
            255,
        ).astype(np.uint8)
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            CONVERGENCE_OUTLINE_COLOR,
            CONVERGENCE_OUTLINE_THICKNESS,
        )

    analysis_output = {
        "danger_regions": danger_boxes,
        "dangerous_cells": critical_cells,
        "convergence_cells": converging_cells,
        "person_count": 0,
    }

    if detection_results and len(detection_results) > 0 and detection_results[0].boxes:
        analysis_output["person_count"] = sum(
            1 for det in detection_results[0].boxes.cls if int(det) == 0
        )

    return annotated_frame, analysis_output


def process_cpu(
    prev_gray_frame: np.ndarray,
    current_frame_color: np.ndarray,
    proc_width: int,
    proc_height: int,
    grid_rows: int,
    grid_cols: int,
    cell_width: int,
    cell_height: int,
    detection_results: Any,
) -> tuple[np.ndarray, Dict[str, Any]]:
    height, width = proc_height, proc_width

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_frame,
        cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2GRAY),
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )

    flow_x_smooth = cv2.GaussianBlur(flow[..., 0], SMOOTHING_KERNEL, 0)
    flow_y_smooth = cv2.GaussianBlur(flow[..., 1], SMOOTHING_KERNEL, 0)
    smoothed_flow = np.stack((flow_x_smooth, flow_y_smooth), axis=-1)

    flow_x = smoothed_flow[..., 0]
    flow_y = smoothed_flow[..., 1]

    if height < 2 or width < 2:
        divergence = np.zeros_like(flow_x)
        curl = np.zeros_like(flow_y)
    else:
        dy_x, dx_x = np.gradient(flow_x)
        dy_y, dx_y = np.gradient(flow_y)
        divergence = dx_x + dy_y
        curl = dx_y - dy_x

    magnitude = np.sqrt(flow_x**2 + flow_y**2)

    mask_div = divergence > DIVERGENCE_THRESH
    mask_crl = np.abs(curl) > CURL_THRESH
    mask_mag = magnitude > MAGNITUDE_THRESH
    mask_critical = (mask_div | mask_crl) & mask_mag
    mask_critical_uint8 = mask_critical.astype(np.uint8) * 255
    mask_critical_morphed = cv2.morphologyEx(
        mask_critical_uint8, cv2.MORPH_OPEN, MORPH_STRUCT_ELEMENT
    )
    mask_critical_morphed = cv2.morphologyEx(
        mask_critical_morphed, cv2.MORPH_CLOSE, MORPH_STRUCT_ELEMENT
    )

    contours, _ = cv2.findContours(
        mask_critical_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    danger_boxes = []
    critical_centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 50:
            danger_boxes.append({"x": x, "y": y, "w": w, "h": h})
            critical_centers.append((x + w // 2, y + h // 2))

    critical_cells = []
    converging_cells = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * cell_height
            y2 = min(height, y1 + cell_height)
            x1 = c * cell_width
            x2 = min(width, x1 + cell_width)

            if y1 < y2 and x1 < x2:
                for center_x, center_y in critical_centers:
                    if x1 <= center_x < x2 and y1 <= center_y < y2:
                        critical_cells.append({"row": r, "col": c})
                        break

                cell_divergence = divergence[y1:y2, x1:x2]
                if (
                    cell_divergence.size > 0
                    and np.min(cell_divergence) < CONVERGENCE_THRESH
                ):
                    converging_cells.append({"row": r, "col": c})

    annotated_frame = current_frame_color.copy()

    annotated_frame = draw_boxes(annotated_frame, detection_results)

    for box in danger_boxes:
        cv2.rectangle(
            annotated_frame,
            (box["x"], box["y"]),
            (box["x"] + box["w"], box["y"] + box["h"]),
            DANGER_BOX_COLOR,
            DANGER_BOX_THICKNESS,
        )

    for i in range(1, grid_cols):
        x = i * cell_width
        cv2.line(
            annotated_frame, (x, 0), (x, height), GRID_LINE_COLOR, GRID_LINE_THICKNESS
        )
    cv2.line(
        annotated_frame,
        (width - 1, 0),
        (width - 1, height),
        GRID_LINE_COLOR,
        GRID_LINE_THICKNESS,
    )

    for i in range(1, grid_rows):
        y = i * cell_height
        cv2.line(
            annotated_frame, (0, y), (width, y), GRID_LINE_COLOR, GRID_LINE_THICKNESS
        )
    cv2.line(
        annotated_frame,
        (0, height - 1),
        (width, height - 1),
        GRID_LINE_COLOR,
        GRID_LINE_THICKNESS,
    )

    for cell_idx in critical_cells:
        r, c = cell_idx["row"], cell_idx["col"]
        x1 = c * cell_width
        y1 = r * cell_height
        x2 = min(width, x1 + cell_width)
        y2 = min(height, y1 + cell_height)
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            DANGER_CELL_COLOR,
            DANGER_CELL_THICKNESS,
        )

    for cell_idx in converging_cells:
        r, c = cell_idx["row"], cell_idx["col"]
        y1 = r * cell_height
        y2 = min(height, y1 + cell_height)
        x1 = c * cell_width
        x2 = min(width, x1 + cell_width)
        annotated_frame[y1:y2, x1:x2] = np.clip(
            annotated_frame[y1:y2, x1:x2].astype(np.float32) + CONVERGENCE_TINT_COLOR,
            0,
            255,
        ).astype(np.uint8)
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            CONVERGENCE_OUTLINE_COLOR,
            CONVERGENCE_OUTLINE_THICKNESS,
        )

    analysis_output = {
        "danger_regions": danger_boxes,
        "dangerous_cells": critical_cells,
        "convergence_cells": converging_cells,
        "person_count": 0,
    }

    if detection_results and len(detection_results) > 0 and detection_results[0].boxes:
        analysis_output["person_count"] = sum(
            1 for det in detection_results[0].boxes.cls if int(det) == 0
        )

    return annotated_frame, analysis_output


def load_image(file_content: bytes) -> np.ndarray | None:
    np_arr = np.frombuffer(file_content, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def draw_boxes(frame: np.ndarray, results: Any) -> np.ndarray:
    annotated_frame = frame.copy()
    if results and len(results) > 0 and results[0].boxes:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated_frame


@router.post("/analyze_frame_stateless/")
async def analyze_frame_endpoint(
    prev_frame: UploadFile = File(..., description="Previous frame image file"),
    current_frame: UploadFile = File(..., description="Current frame image file"),
) -> JSONResponse:
    try:
        prev_content = await prev_frame.read()
        current_content = await current_frame.read()

        prev_img = load_image(prev_content)
        current_img = load_image(current_content)

        if prev_img is None or current_img is None:
            raise HTTPException(
                status_code=400, detail="Could not decode one or both images."
            )

        current_height, current_width = current_img.shape[:2]
        if current_width > MAX_DIMENSION:
            proc_width = MAX_DIMENSION
            proc_height = int(current_height * (MAX_DIMENSION / current_width))
        else:
            proc_width = current_width
            proc_height = current_height

        current_img_resized = cv2.resize(current_img, (proc_width, proc_height))

        detection_results = None
        if model:
            try:
                detection_results = model(
                    current_img_resized, classes=[0], verbose=False
                )
            except Exception as e:
                logging.error(f"YOLO detection failed: {e}")
                detection_results = None
        else:
            logging.warning("YOLO model not loaded. Skipping detection.")

        prev_gray_resized = cv2.resize(
            cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY),
            (proc_width, proc_height),
        )

        aspect_ratio = proc_width / proc_height
        est_rows = round(math.sqrt(GRID_CELL_COUNT / aspect_ratio))
        est_cols = round(est_rows * aspect_ratio)

        grid_rows = max(1, est_rows)
        grid_cols = max(1, est_cols)

        current_cells = grid_rows * grid_cols
        if current_cells < GRID_CELL_COUNT:
            if aspect_ratio > 1:
                grid_cols = max(1, grid_cols + 1)
            else:
                grid_rows = max(1, grid_rows + 1)

        cell_width = max(1, proc_width // grid_cols)
        cell_height = max(1, proc_height // grid_rows)

        if gpu_available:
            try:
                prev_gray_gpu = cv2.cuda_GpuMat()
                prev_gray_gpu.upload(prev_gray_resized)

                current_img_gpu = cv2.cuda_GpuMat()
                current_img_gpu.upload(current_img_resized)

                annotated_frame, analysis_output = process_gpu(
                    prev_gray_gpu,
                    current_img_gpu,
                    proc_width,
                    proc_height,
                    grid_rows,
                    grid_cols,
                    cell_width,
                    cell_height,
                    detection_results,
                )
            except Exception as e:
                logging.error(f"GPU analysis failed: {e}. Falling back to CPU.")
                annotated_frame, analysis_output = process_cpu(
                    prev_gray_resized,
                    current_img_resized,
                    proc_width,
                    proc_height,
                    grid_rows,
                    grid_cols,
                    cell_width,
                    cell_height,
                    detection_results,
                )
        else:
            annotated_frame, analysis_output = process_cpu(
                prev_gray_resized,
                current_img_resized,
                proc_width,
                proc_height,
                grid_rows,
                grid_cols,
                cell_width,
                cell_height,
                detection_results,
            )

        is_success, buffer = cv2.imencode(".jpg", annotated_frame)
        if not is_success:
            raise HTTPException(
                status_code=500, detail="Could not encode processed image."
            )

        analysis_output["annotated_frame_base64"] = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        return JSONResponse(content=analysis_output)

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
