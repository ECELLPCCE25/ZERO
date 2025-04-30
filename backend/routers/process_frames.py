import logging
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO
import base64
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import math
import matplotlib.pyplot as plt  # Import for plotting
import matplotlib.cm as cm  # Import for colormaps
import matplotlib.colors as mcolors  # Import for color handling
import io  # Import for handling byte streams

router = APIRouter()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    model = YOLO("yolov8n.pt")
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    model = None

# --- Smoothing Parameters ---
# Kernel size for Gaussian blur (increase slightly for better region detection)
SPATIAL_SMOOTHING_KERNEL_SIZE = (5, 5)

# --- Optical Flow Analysis Parameters ---
CONVERGENCE_THRESH = (
    -0.6
)  # Threshold for detecting significant convergence (negative divergence)
DIVERGENCE_THRESH = (
    0.6  # Threshold for detecting significant divergence (positive divergence)
)
CURL_THRESH = 0.2  # Threshold for detecting significant curl (swirling)
MAGNITUDE_THRESH = 0.8  # Threshold for significant movement magnitude

# --- Visualization Parameters ---
DANGER_BOX_COLOR = (0, 80, 255)  # Orange/Red color for danger rectangles (BGR)
DANGER_BOX_THICKNESS = 2  # Thickness of the danger rectangle border

GRID_CELL_COUNT = 15  # Desired approximate total number of grid cells
GRID_LINE_COLOR = (200, 200, 200)  # Light grey for grid lines (BGR)
GRID_LINE_THICKNESS = 1  # Thickness of grid lines

DANGER_CELL_COLOR = (
    0,
    0,
    255,
)  # Red color for highlighting generally dangerous cells (BGR)
DANGER_CELL_THICKNESS = 2  # Thickness of the highlight border

# BGR color to add for blue tint (e.g., 100 Blue, 0 Green, 0 Red)
CONVERGENCE_TINT_COLOR = (100, 0, 0)
CONVERGENCE_OUTLINE_COLOR = (255, 255, 255)  # White color for the outline (BGR)
CONVERGENCE_OUTLINE_THICKNESS = 1  # Thin thickness for the outline

PREDICTION_COLOR = (0, 165, 255)  # Standard Orange color for predicted locations (BGR)
PREDICTION_RADIUS = 5  # Radius of the circle marker for prediction
PREDICTION_STEPS = (
    1  # Number of frames to predict into the future (e.g., 1 for next frame)
)

# --- Processing Parameters ---
MAX_DIMENSION = 640  # Maximum width for displayed frames

# Kernel for morphological operations (connecting nearby dangerous pixels)
MORPH_STRUCT_ELEMENT = np.ones((5, 5), np.uint8)

# --- Quiver Plot Parameters ---
QUIVER_STEP = 16  # Adjust this value to change the density of arrows
MAX_ARROW_DISPLAY_LENGTH = 50.0  # Desired max length of the longest arrow in plot units


gpu_available = False
try:
    # Check if OpenCV CUDA is available and can be initialized
    # Creating a GpuMat or checking device count can trigger initialization errors
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        cv2.cuda.GpuMat()  # Attempt to create a GpuMat to ensure initialization works
        logging.info("OpenCV CUDA device detected and initialized.")
        gpu_available = True
    else:
        logging.warning(
            "OpenCV CUDA is enabled but no devices found. Falling back to CPU."
        )

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


def generate_quiver_plot_base64(
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    width: int,
    height: int,
    step: int,
    max_display_length: float,
) -> str | None:
    """
    Generates a base64 encoded Matplotlib quiver plot of the optical flow.
    """
    try:
        y_coords, x_coords = np.mgrid[
            step // 2 : height : step, step // 2 : width : step
        ]
        u_values = flow_u[y_coords, x_coords]
        v_values = flow_v[y_coords, x_coords]

        # Calculate magnitudes for normalization
        magnitudes = np.sqrt(u_values**2 + v_values**2)
        max_magnitude = np.max(magnitudes)

        # Normalize vectors for consistent arrow length display
        if max_magnitude > 1e-6:  # Avoid division by zero or very small numbers
            u_normalized = u_values / max_magnitude
            v_normalized = v_values / max_magnitude
        else:
            u_normalized = u_values
            v_normalized = v_values

        plt.ioff()  # Turn off interactive mode for plot generation
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100), dpi=100
        )  # Adjust figure size

        # Quiver plot - note: origin='upper' to match image coordinates
        # The 'scale' parameter controls arrow length. Larger scale means shorter arrows for the same unit length.
        # We want max_display_length to correspond to the max_magnitude after normalization (which is 1).
        # So, scale should be such that arrow length 1 in data units corresponds to max_display_length in plot units.
        # Matplotlib's scale is data units per arrow length unit.
        # If we normalize vectors to max_magnitude=1, we want arrow length 1 to map to max_display_length.
        # Quiver scales arrows by their data units. An arrow (u,v) has length sqrt(u^2+v^2).
        # After normalization, max length is 1. We want this to display as max_display_length.
        # Let's use scale = 1 / max_display_length, assuming display units are relative to axis limits (which quiver does).
        # Quiver is tricky with scaling; let's use angles and magnitude for color and a fixed display length for clarity.
        # We can scale the vectors directly to control display length relative to the grid step.
        # Let's try scaling so max arrow length is proportional to the step, e.g., 0.8 * step
        max_arrow_data_length = (
            0.8 * step
        )  # Max arrow length in data coordinates (pixels)

        if max_magnitude > 1e-6:
            u_display = (
                u_values
                / magnitudes
                * max_arrow_data_length
                * (magnitudes / max_magnitude)
            )
            v_display = (
                v_values
                / magnitudes
                * max_arrow_data_length
                * (magnitudes / max_magnitude)
            )
            # Use angles for color
            angles = np.arctan2(
                -v_values, -u_values
            )  # Note: y-axis is inverted in images, flow is opposite direction
            angles = np.rad2deg(angles)  # Convert to degrees
            # Map angles to hue (0-180 for OpenCV's HSV hue, 0-360 for Matplotlib's colormaps)
            norm = mcolors.Normalize(vmin=0, vmax=360)
            cmap = cm.hsv  # HSV colormap
            colors = cmap(norm((angles + 360) % 360))  # Ensure angles are positive
        else:
            u_display = np.zeros_like(u_values)
            v_display = np.zeros_like(v_values)
            colors = "blue"  # Default color if no flow

        ax.quiver(
            x_coords,
            y_coords,
            u_display,
            v_display,
            color=colors,
            angles="xy",  # Interpret U,V as x,y components
            scale_units="xy",  # Match scaling to x,y units
            scale=1,  # No additional scaling needed if vectors are already scaled
            pivot="mid",  # Pivot arrow from the center
        )

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
        ax.set_aspect("equal", adjustable="box")  # Maintain aspect ratio
        ax.axis("off")  # Hide axes
        ax.set_title("Optical Flow Quiver Plot")

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)  # Close the figure to free memory

        # Encode to base64
        quiver_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return quiver_base64

    except Exception as e:
        logging.error(f"Error generating quiver plot: {e}")
        return None


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
) -> tuple[np.ndarray, Dict[str, Any], List[Dict[str, int]]]:
    height, width = proc_height, proc_width

    stream = cv2.cuda_Stream()
    current_gray_gpu = cv2.cuda.cvtColor(
        current_frame_gpu, cv2.COLOR_BGR2GRAY, stream=stream
    )

    flow_calc = cv2.cuda_FarnebackOpticalFlow.create()
    flow_data_gpu = flow_calc.calc(
        prev_frame_gpu, current_gray_gpu, None, stream=stream
    )

    blur_filter_gpu = cv2.cuda.createGaussianFilter(
        cv2.CV_32FC2, cv2.CV_32FC2, SPATIAL_SMOOTHING_KERNEL_SIZE, 0
    )
    # Note: apply might return a new GpuMat if output is None
    smoothed_flow_gpu = blur_filter_gpu.apply(flow_data_gpu, stream=stream)

    # Download smoothed flow data to CPU for divergence/curl/magnitude/prediction
    smoothed_flow_cpu = smoothed_flow_gpu.download(stream=stream)
    stream.waitForCompletion()  # Wait for download to complete

    flow_x = smoothed_flow_cpu[..., 0]
    flow_y = smoothed_flow_cpu[..., 1]

    # Calculate divergence and curl
    if height < 2 or width < 2:  # Handle edge case for very small frames
        divergence = np.zeros_like(flow_x)
        curl = np.zeros_like(flow_y)
    else:
        # Use central differences for gradient where possible, pad edges
        # Using numpy.gradient is convenient but might be slow on large data
        # For performance, consider optimizing gradient calculation or using GPU where available (less common for gradient)
        dy_x, dx_x = np.gradient(flow_x)
        dy_y, dx_y = np.gradient(flow_y)
        divergence = dx_x + dy_y
        curl = dx_y - dy_x  # Standard curl definition in 2D (scalar)

    magnitude = np.sqrt(flow_x**2 + flow_y**2)

    # --- Danger Region Detection (using Divergence, Curl, AND Magnitude) ---
    mask_div = divergence > DIVERGENCE_THRESH
    mask_crl = np.abs(curl) > CURL_THRESH
    mask_mag = magnitude > MAGNITUDE_THRESH  # New mask for significant movement

    # Combine masks: A pixel is considered potentially dangerous if it has
    # significant divergence OR significant curl AND significant magnitude.
    mask_critical = (mask_div | mask_crl) & mask_mag
    mask_critical_uint8 = mask_critical.astype(np.uint8) * 255

    # Apply morphological operations to connect nearby dangerous pixels
    mask_critical_morphed = cv2.morphologyEx(
        mask_critical_uint8, cv2.MORPH_OPEN, MORPH_STRUCT_ELEMENT
    )
    mask_critical_morphed = cv2.morphologyEx(
        mask_critical_morphed, cv2.MORPH_CLOSE, MORPH_STRUCT_ELEMENT
    )

    # Find contours on the morphed danger mask
    contours, _ = cv2.findContours(
        mask_critical_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    danger_boxes = []
    critical_centers = []  # Centers of the detected critical regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Minimum area threshold for a valid danger region (tune this)
        if w * h > 50:
            danger_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
            critical_centers.append((x + w // 2, y + h // 2))

    critical_cells = []  # Cells containing critical region centers
    converging_cells = []  # Cells with min divergence below CONVERGENCE_THRESH
    predicted_points = []  # Predicted locations for convergence cells

    # Calculate cell dimensions based on the determined grid size
    cell_width = max(1, width // grid_cols)
    cell_height = max(1, height // grid_rows)

    for r in range(grid_rows):
        for c in range(grid_cols):
            cy1 = r * cell_height
            cy2 = min(height, cy1 + cell_height)  # Clamp bounds
            cx1 = c * cell_width
            cx2 = min(width, cx1 + cell_width)  # Clamp bounds

            # Check if any critical region centers fall into this cell
            is_cell_critical = False
            for center_x, center_y in critical_centers:
                if cx1 <= center_x < cx2 and cy1 <= center_y < cy2:
                    critical_cells.append({"row": r, "col": c})
                    is_cell_critical = True
                    break  # No need to check other centers for this cell

            # Check if the minimum divergence in this cell meets the convergence threshold
            if cy1 < cy2 and cx1 < cx2:  # Ensure the slice is valid
                cell_divergence = divergence[cy1:cy2, cx1:cx2]
                if cell_divergence.size > 0:
                    # If the minimum divergence is below the threshold (more negative)
                    if np.min(cell_divergence) < CONVERGENCE_THRESH:
                        converging_cells.append({"row": r, "col": c})

                        # --- Prediction for Convergence Cells ---
                        # Get the flow vector at the center of the convergence cell
                        # Note: This is a simplification, a more robust method might average flow in the cell
                        center_y_flow = min(int(cy1 + cell_height // 2), height - 1)
                        center_x_flow = min(int(cx1 + cell_width // 2), width - 1)

                        # Ensure indices are within bounds before accessing flow
                        if 0 <= center_y_flow < height and 0 <= center_x_flow < width:
                            flow_at_center_u = flow_x[center_y_flow, center_x_flow]
                            flow_at_center_v = flow_y[center_y_flow, center_x_flow]

                            # Predict the future center position
                            predicted_center_x = (
                                cx1
                                + cell_width // 2
                                + flow_at_center_u * PREDICTION_STEPS
                            )
                            predicted_center_y = (
                                cy1
                                + cell_height // 2
                                + flow_at_center_v * PREDICTION_STEPS
                            )

                            # Clamp predicted positions to stay within frame bounds
                            predicted_center_x = np.clip(
                                predicted_center_x, 0, width - 1
                            )
                            predicted_center_y = np.clip(
                                predicted_center_y, 0, height - 1
                            )

                            predicted_points.append(
                                {
                                    "x": int(predicted_center_x),
                                    "y": int(predicted_center_y),
                                }
                            )

    # Download current frame for drawing
    current_frame_cpu = current_frame_gpu.download(stream=stream)
    stream.waitForCompletion()

    annotated_frame = current_frame_cpu.copy()

    # Draw YOLO detection boxes
    annotated_frame = draw_boxes(annotated_frame, detection_results)

    # Draw Danger Rectangles
    for box in danger_boxes:
        cv2.rectangle(
            annotated_frame,
            (box["x"], box["y"]),
            (box["x"] + box["w"], box["y"] + box["h"]),
            DANGER_BOX_COLOR,
            DANGER_BOX_THICKNESS,
        )

    # Draw Grid Lines
    for i in range(1, grid_cols):
        x = i * cell_width
        cv2.line(
            annotated_frame, (x, 0), (x, height), GRID_LINE_COLOR, GRID_LINE_THICKNESS
        )
    # Draw the rightmost vertical line explicitly if it doesn't align perfectly
    if width > 0:  # Avoid drawing if width is zero
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
    # Draw the bottom horizontal line explicitly if it doesn't align perfectly
    if height > 0:  # Avoid drawing if height is zero
        cv2.line(
            annotated_frame,
            (0, height - 1),
            (width, height - 1),
            GRID_LINE_COLOR,
            GRID_LINE_THICKNESS,
        )

    # Highlight generally dangerous cells (red border)
    for cell_idx in critical_cells:
        r, c = cell_idx["row"], cell_idx["col"]
        cx1 = c * cell_width
        cy1 = r * cell_height
        cx2 = min(width, cx1 + cell_width)  # Clamp bounds
        cy2 = min(height, cy1 + cell_height)  # Clamp bounds
        cv2.rectangle(
            annotated_frame,
            (cx1, cy1),
            (cx2, cy2),
            DANGER_CELL_COLOR,
            DANGER_CELL_THICKNESS,
        )

    # Highlight converging cells (blue tint and white outline)
    for cell_idx in converging_cells:
        r, c = cell_idx["row"], cell_idx["col"]
        cy1 = r * cell_height
        cy2 = min(height, cy1 + cell_height)  # Clamp bounds
        cx1 = c * cell_width
        cx2 = min(width, cx1 + cell_width)  # Clamp bounds

        # Apply tint - ensure slice is valid
        if cy1 < cy2 and cx1 < cx2:
            annotated_frame[cy1:cy2, cx1:cx2] = np.clip(
                annotated_frame[cy1:cy2, cx1:cx2].astype(np.float32)
                + CONVERGENCE_TINT_COLOR,
                0,
                255,
            ).astype(np.uint8)
            # Draw outline
            cv2.rectangle(
                annotated_frame,
                (cx1, cy1),
                (cx2, cy2),
                CONVERGENCE_OUTLINE_COLOR,
                CONVERGENCE_OUTLINE_THICKNESS,
            )

    # Draw Predicted Points
    for p in predicted_points:
        cv2.circle(
            annotated_frame, (p["x"], p["y"]), PREDICTION_RADIUS, PREDICTION_COLOR, -1
        )  # -1 fills the circle

    analysis_output = {
        "danger_regions": danger_boxes,
        "dangerous_cells": critical_cells,  # These are cells overlapping danger_regions
        "convergence_cells": converging_cells,  # These are cells meeting the convergence threshold
        "predicted_convergence_points": predicted_points,
        "person_count": 0,
    }

    # Count persons if detection results are available
    if (
        detection_results
        and len(detection_results) > 0
        and detection_results[0].boxes is not None
    ):
        # YOLO class ID for 'person' is typically 0
        analysis_output["person_count"] = sum(
            1
            for det_cls in detection_results[0].boxes.cls.cpu().numpy()
            if int(det_cls) == 0
        )

    return annotated_frame, analysis_output, smoothed_flow_cpu


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
) -> tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    height, width = proc_height, proc_width

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_frame,
        cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2GRAY),
        None,
        0.5,  # pyr_scale
        3,  # levels
        15,  # winsize
        3,  # iterations
        5,  # poly_n
        1.2,  # poly_sigma
        0,  # flags
    )

    # Apply Spatial Smoothing
    flow_x_smooth = cv2.GaussianBlur(flow[..., 0], SPATIAL_SMOOTHING_KERNEL_SIZE, 0)
    flow_y_smooth = cv2.GaussianBlur(flow[..., 1], SPATIAL_SMOOTHING_KERNEL_SIZE, 0)
    smoothed_flow = np.stack((flow_x_smooth, flow_y_smooth), axis=-1)

    flow_x = smoothed_flow[..., 0]
    flow_y = smoothed_flow[..., 1]

    # Calculate divergence and curl
    if height < 2 or width < 2:  # Handle edge case for very small frames
        divergence = np.zeros_like(flow_x)
        curl = np.zeros_like(flow_y)
    else:
        # Using numpy.gradient
        dy_x, dx_x = np.gradient(flow_x)
        dy_y, dx_y = np.gradient(flow_y)
        divergence = dx_x + dy_y
        curl = (
            dx_y - dy_y
        )  # This seems incorrect, should be dx_y - dy_x based on standard definition. Correcting.
        # curl = dx_y - dy_x # Standard definition

        # Double-checking the curl calculation in the second script: dv_dx - du_dy. Yes, this is dx_y - dy_x
        # So, fixing the CPU curl calculation here to match the second script's logic and standard definition.
        curl = dx_y - dy_x  # Corrected Curl calculation

    magnitude = np.sqrt(flow_x**2 + flow_y**2)

    # --- Danger Region Detection (using Divergence, Curl, AND Magnitude) ---
    mask_div = divergence > DIVERGENCE_THRESH
    mask_crl = np.abs(curl) > CURL_THRESH
    mask_mag = magnitude > MAGNITUDE_THRESH  # New mask for significant movement

    # Combine masks: A pixel is considered potentially dangerous if it has
    # significant divergence OR significant curl AND significant magnitude.
    mask_critical = (mask_div | mask_crl) & mask_mag
    mask_critical_uint8 = mask_critical.astype(np.uint8) * 255

    # Apply morphological operations to connect nearby dangerous pixels
    mask_critical_morphed = cv2.morphologyEx(
        mask_critical_uint8, cv2.MORPH_OPEN, MORPH_STRUCT_ELEMENT
    )
    mask_critical_morphed = cv2.morphologyEx(
        mask_critical_morphed, cv2.MORPH_CLOSE, MORPH_STRUCT_ELEMENT
    )

    # Find contours on the morphed danger mask
    contours, _ = cv2.findContours(
        mask_critical_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    danger_boxes = []
    critical_centers = []  # Centers of the detected critical regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Minimum area threshold for a valid danger region (tune this)
        if w * h > 50:
            danger_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
            critical_centers.append((x + w // 2, y + h // 2))

    critical_cells = []  # Cells containing critical region centers
    converging_cells = []  # Cells with min divergence below CONVERGENCE_THRESH
    predicted_points = []  # Predicted locations for convergence cells

    # Calculate cell dimensions based on the determined grid size
    cell_width = max(1, width // grid_cols)
    cell_height = max(1, height // grid_rows)

    for r in range(grid_rows):
        for c in range(grid_cols):
            cy1 = r * cell_height
            cy2 = min(height, cy1 + cell_height)  # Clamp bounds
            cx1 = c * cell_width
            cx2 = min(width, cx1 + cell_width)  # Clamp bounds

            # Check if any critical region centers fall into this cell
            is_cell_critical = False
            for center_x, center_y in critical_centers:
                if cx1 <= center_x < cx2 and cy1 <= center_y < cy2:
                    critical_cells.append({"row": r, "col": c})
                    is_cell_critical = True
                    break  # No need to check other centers for this cell

            # Check if the minimum divergence in this cell meets the convergence threshold
            if cy1 < cy2 and cx1 < cx2:  # Ensure the slice is valid
                cell_divergence = divergence[cy1:cy2, cx1:cx2]
                if cell_divergence.size > 0:
                    # If the minimum divergence is below the threshold (more negative)
                    if np.min(cell_divergence) < CONVERGENCE_THRESH:
                        converging_cells.append({"row": r, "col": c})

                        # --- Prediction for Convergence Cells ---
                        # Get the flow vector at the center of the convergence cell
                        # Note: This is a simplification, a more robust method might average flow in the cell
                        center_y_flow = min(int(cy1 + cell_height // 2), height - 1)
                        center_x_flow = min(int(cx1 + cell_width // 2), width - 1)

                        # Ensure indices are within bounds before accessing flow
                        if 0 <= center_y_flow < height and 0 <= center_x_flow < width:
                            flow_at_center_u = flow_x[center_y_flow, center_x_flow]
                            flow_at_center_v = flow_y[center_y_flow, center_x_flow]

                            # Predict the future center position
                            predicted_center_x = (
                                cx1
                                + cell_width // 2
                                + flow_at_center_u * PREDICTION_STEPS
                            )
                            predicted_center_y = (
                                cy1
                                + cell_height // 2
                                + flow_at_center_v * PREDICTION_STEPS
                            )

                            # Clamp predicted positions to stay within frame bounds
                            predicted_center_x = np.clip(
                                predicted_center_x, 0, width - 1
                            )
                            predicted_center_y = np.clip(
                                predicted_center_y, 0, height - 1
                            )

                            predicted_points.append(
                                {
                                    "x": int(predicted_center_x),
                                    "y": int(predicted_center_y),
                                }
                            )

    annotated_frame = current_frame_color.copy()

    # Draw YOLO detection boxes
    annotated_frame = draw_boxes(annotated_frame, detection_results)

    # Draw Danger Rectangles
    for box in danger_boxes:
        cv2.rectangle(
            annotated_frame,
            (box["x"], box["y"]),
            (box["x"] + box["w"], box["y"] + box["h"]),
            DANGER_BOX_COLOR,
            DANGER_BOX_THICKNESS,
        )

    # Draw Grid Lines
    for i in range(1, grid_cols):
        x = i * cell_width
        cv2.line(
            annotated_frame, (x, 0), (x, height), GRID_LINE_COLOR, GRID_LINE_THICKNESS
        )
    # Draw the rightmost vertical line explicitly if it doesn't align perfectly
    if width > 0:  # Avoid drawing if width is zero
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
    # Draw the bottom horizontal line explicitly if it doesn't align perfectly
    if height > 0:  # Avoid drawing if height is zero
        cv2.line(
            annotated_frame,
            (0, height - 1),
            (width, height - 1),
            GRID_LINE_COLOR,
            GRID_LINE_THICKNESS,
        )

    # Highlight generally dangerous cells (red border)
    for cell_idx in critical_cells:
        r, c = cell_idx["row"], cell_idx["col"]
        cx1 = c * cell_width
        cy1 = r * cell_height
        cx2 = min(width, cx1 + cell_width)  # Clamp bounds
        cy2 = min(height, cy1 + cell_height)  # Clamp bounds
        cv2.rectangle(
            annotated_frame,
            (cx1, cy1),
            (cx2, cy2),
            DANGER_CELL_COLOR,
            DANGER_CELL_THICKNESS,
        )

    # Highlight converging cells (blue tint and white outline)
    for cell_idx in converging_cells:
        r, c = cell_idx["row"], cell_idx["col"]
        cy1 = r * cell_height
        cy2 = min(height, cy1 + cell_height)  # Clamp bounds
        cx1 = c * cell_width
        cx2 = min(width, cx1 + cell_width)  # Clamp bounds

        # Apply tint - ensure slice is valid
        if cy1 < cy2 and cx1 < cx2:
            annotated_frame[cy1:cy2, cx1:cx2] = np.clip(
                annotated_frame[cy1:cy2, cx1:cx2].astype(np.float32)
                + CONVERGENCE_TINT_COLOR,
                0,
                255,
            ).astype(np.uint8)
            # Draw outline
            cv2.rectangle(
                annotated_frame,
                (cx1, cy1),
                (cx2, cy2),
                CONVERGENCE_OUTLINE_COLOR,
                CONVERGENCE_OUTLINE_THICKNESS,
            )

    # Draw Predicted Points
    for p in predicted_points:
        cv2.circle(
            annotated_frame, (p["x"], p["y"]), PREDICTION_RADIUS, PREDICTION_COLOR, -1
        )  # -1 fills the circle

    analysis_output = {
        "danger_regions": danger_boxes,
        "dangerous_cells": critical_cells,  # These are cells overlapping danger_regions
        "convergence_cells": converging_cells,  # These are cells meeting the convergence threshold
        "predicted_convergence_points": predicted_points,
        "person_count": 0,
    }

    # Count persons if detection results are available
    if (
        detection_results
        and len(detection_results) > 0
        and detection_results[0].boxes is not None
    ):
        # YOLO class ID for 'person' is typically 0
        analysis_output["person_count"] = sum(
            1
            for det_cls in detection_results[0].boxes.cls.cpu().numpy()
            if int(det_cls) == 0
        )

    return annotated_frame, analysis_output, smoothed_flow


def load_image(file_content: bytes) -> np.ndarray | None:
    """Loads image from bytes."""
    try:
        np_arr = np.frombuffer(file_content, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logging.error("cv2.imdecode failed.")
        return frame
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return None


def draw_boxes(frame: np.ndarray, results: Any) -> np.ndarray:
    """Draws bounding boxes from YOLO results onto the frame."""
    # Ensure results is not None and has expected structure
    if results is None or len(results) == 0 or results[0].boxes is None:
        return frame  # Return original frame if no results or boxes

    annotated_frame = frame.copy()
    try:
        # Ensure we handle potential empty boxes.xyxy
        if results[0].boxes.xyxy is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                # Ensure coordinates are within frame bounds before drawing
                height, width = annotated_frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)
                if x2 > x1 and y2 > y1:  # Only draw valid rectangles
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except Exception as e:
        logging.error(f"Error drawing detection boxes: {e}")

    return annotated_frame


@router.post("/analyze_frame_stateless/")
async def analyze_frame_endpoint(
    prev_frame: UploadFile = File(..., description="Previous frame image file"),
    current_frame: UploadFile = File(..., description="Current frame image file"),
) -> JSONResponse:
    """
    Analyzes two consecutive frames using optical flow and YOLO detection,
    identifies danger/convergence zones, predicts movement, and returns
    an annotated frame and analysis data, including a quiver plot.
    """
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

        # Determine processing dimensions while maintaining aspect ratio and respecting MAX_DIMENSION
        if current_width > MAX_DIMENSION:
            proc_width = MAX_DIMENSION
            proc_height = int(current_height * (MAX_DIMENSION / current_width))
            # Ensure height is at least 1
            proc_height = max(1, proc_height)
        else:
            proc_width = current_width
            proc_height = current_height

        # Resize current frame for processing and display
        current_img_resized = cv2.resize(current_img, (proc_width, proc_height))

        # Perform YOLO detection on the resized current frame if model is loaded
        detection_results = None
        if model:
            try:
                # Filter for class 0 (person)
                detection_results = model(
                    current_img_resized, classes=[0], verbose=False
                )
            except Exception as e:
                logging.error(f"YOLO detection failed: {e}")
                detection_results = None
        else:
            logging.warning("YOLO model not loaded. Skipping detection.")

        # Resize previous frame to grayscale for optical flow
        # Ensure prev_img is not None before processing
        if prev_img is None:
            # This case should ideally be caught by the initial None check, but good practice
            logging.error("Previous image is None before grayscale/resize.")
            raise HTTPException(
                status_code=500, detail="Internal error processing previous frame."
            )

        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        prev_gray_resized = cv2.resize(
            prev_gray,
            (proc_width, proc_height),
        )

        # --- Calculate Grid Dimensions ---
        # Ensure dimensions are positive before calculation
        if proc_width <= 0 or proc_height <= 0:
            raise HTTPException(
                status_code=400, detail="Invalid image dimensions after resize."
            )

        aspect_ratio = proc_width / proc_height
        # Estimate rows and columns to get close to GRID_CELL_COUNT while maintaining aspect ratio
        est_rows = round(math.sqrt(GRID_CELL_COUNT / aspect_ratio))
        est_cols = round(est_rows * aspect_ratio)

        # Ensure minimum 1 row and 1 column
        grid_rows = max(1, est_rows)
        grid_cols = max(1, est_cols)

        # Simple adjustment if cell count is significantly off (optional, depends on desired grid behavior)
        # This part of the logic can be complex to optimize perfectly; the current approach is a heuristic.
        # For simplicity, let's use the calculated grid_rows and grid_cols directly after ensuring minimums.

        # Recalculate cell dimensions based on the determined grid size
        cell_width = max(1, proc_width // grid_cols)
        cell_height = max(1, proc_height // grid_rows)

        # Perform analysis using GPU or CPU
        smoothed_flow = None
        if gpu_available:
            try:
                # Upload grayscale previous frame and color current frame to GPU
                prev_gray_gpu = cv2.cuda_GpuMat()
                prev_gray_gpu.upload(prev_gray_resized)

                current_img_gpu = cv2.cuda_GpuMat()
                current_img_gpu.upload(current_img_resized)

                # Process on GPU
                annotated_frame, analysis_output, smoothed_flow = process_gpu(
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
                logging.info("GPU analysis completed.")
            except Exception as e:
                logging.error(f"GPU analysis failed: {e}. Falling back to CPU.")
                gpu_available = False  # Mark GPU as failed for this request/future requests if persistent

        if not gpu_available:
            # Process on CPU
            annotated_frame, analysis_output, smoothed_flow = process_cpu(
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
            logging.info("CPU analysis completed.")

        # Generate Quiver Plot
        quiver_base64 = generate_quiver_plot_base64(
            smoothed_flow[..., 0],
            smoothed_flow[..., 1],
            proc_width,
            proc_height,
            QUIVER_STEP,
            MAX_ARROW_DISPLAY_LENGTH,  # Pass max display length parameter
        )
        if quiver_base64:
            analysis_output["quiver_plot_base64"] = quiver_base64
        else:
            analysis_output["quiver_plot_base64"] = None
            logging.warning("Failed to generate quiver plot.")

        # Encode annotated frame to base64
        is_success, buffer = cv2.imencode(".jpg", annotated_frame)
        if not is_success:
            raise HTTPException(
                status_code=500, detail="Could not encode processed image."
            )

        analysis_output["annotated_frame_base64"] = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        return JSONResponse(content=analysis_output)

    except HTTPException as http_exc:
        # Re-raise HTTPException to be handled by FastAPI
        logging.error(f"HTTPException: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}")
        # Return a generic 500 error for unhandled exceptions
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
