import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Constants ---
SPATIAL_SMOOTHING_KERNEL_SIZE = (5, 5)
TEMPORAL_SMOOTHING_FRAMES = 2
QUIVER_STEP = 16
MAX_ARROW_DISPLAY_LENGTH = 50.0
CONVERGENCE_THRESHOLD = -0.6
DIVERGENCE_THRESHOLD = 0.6
CURL_THRESHOLD = 0.2
MAGNITUDE_THRESHOLD = 0.8
DANGER_RECT_COLOR = (0, 80, 255)  # Blue in BGR
DANGER_RECT_THICKNESS = 2
NUM_GRID_CELLS = 15
GRID_COLOR = (200, 200, 200)  # Light gray in BGR
GRID_THICKNESS = 1
GENERAL_DANGER_CELL_COLOR = (0, 0, 255)  # Red in BGR
GENERAL_DANGER_CELL_THICKNESS = 2
CONVERGENCE_THRESHOLD_TINT_COLOR = (100, 0, 0)  # Dark Blue tint in BGR
CONVERGENCE_THRESHOLD_OUTLINE_COLOR = (255, 255, 255)  # White in BGR
CONVERGENCE_THRESHOLD_OUTLINE_THICKNESS = 1
PREDICTION_STEPS = 1
PREDICTION_COLOR = (0, 165, 255)  # Orange in BGR
PREDICTION_RADIUS = 5
PREDICTION_FADE_FRAMES = 30
MAX_WIDTH = 240
MORPH_KERNEL = np.ones((5, 5), np.uint8)


def process_frame(
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
    ax,
):
    """
    Processes a single frame to calculate optical flow, detect dangerous regions,
    and update visualizations.

    Args:
        cap: OpenCV VideoCapture object.
        prev_gray: Previous grayscale frame.
        new_width: Resized width of the frame.
        new_height: Resized height of the frame.
        hsv: HSV image for flow visualization.
        flow_history: Deque storing recent flow fields for temporal smoothing.
        active_predictions: List of active prediction points with fade age.
        grid_rows: Number of grid rows.
        grid_cols: Number of grid columns.
        cell_width: Width of each grid cell.
        cell_height: Height of each grid cell.
        ax: Matplotlib Axes object for the quiver plot.

    Returns:
        A tuple containing:
            - bool: True if the frame was processed successfully, False otherwise.
            - np.ndarray: The current grayscale frame.
            - np.ndarray: The frame with visualizations drawn.
            - np.ndarray: The quiver plot image as a NumPy array.
    """
    ret, frame2 = cap.read()
    if not ret:
        # Attempt to loop the video if it's a file
        if isinstance(cap, cv2.VideoCapture) and cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
            print("Looping video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame2 = cap.read()
            if not ret:
                print("Error looping video.")
                return (
                    False,
                    prev_gray,
                    None,
                    None,
                )  # Return None for display_frame and quiver_img on error
        else:
            return (
                False,
                prev_gray,
                None,
                None,
            )  # Return None for display_frame and quiver_img on error

    # Resize frame
    frame2_resized = cv2.resize(frame2, (new_width, new_height))
    next_gray = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Spatial smoothing
    flow_u_smooth = cv2.GaussianBlur(flow[..., 0], SPATIAL_SMOOTHING_KERNEL_SIZE, 0)
    flow_v_smooth = cv2.GaussianBlur(flow[..., 1], SPATIAL_SMOOTHING_KERNEL_SIZE, 0)
    spatially_smoothed_flow = np.stack((flow_u_smooth, flow_v_smooth), axis=-1)

    # Temporal smoothing
    flow_history.append(spatially_smoothed_flow)
    # Ensure flow_history does not exceed TEMPORAL_SMOOTHING_FRAMES
    while len(flow_history) > TEMPORAL_SMOOTHING_FRAMES:
        flow_history.popleft()
    temporally_smoothed_flow = np.mean(list(flow_history), axis=0)

    flow_u = temporally_smoothed_flow[..., 0]
    flow_v = temporally_smoothed_flow[..., 1]

    # Calculate divergence and curl
    du_dy, du_dx = np.gradient(flow_u)
    dv_dy, dv_dx = np.gradient(flow_v)

    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy

    # Calculate magnitude
    magnitude, _ = cv2.cartToPolar(flow_u, flow_v)

    # Identify dangerous regions based on thresholds
    mask_divergence = divergence > DIVERGENCE_THRESHOLD
    mask_curl = np.abs(curl) > CURL_THRESHOLD
    mask_magnitude = magnitude > MAGNITUDE_THRESHOLD

    # Combine masks to find dangerous areas
    mask_dangerous = (mask_divergence | mask_curl) & mask_magnitude

    # Morphological operations to clean up the mask
    mask_dangerous_uint8 = mask_dangerous.astype(np.uint8) * 255
    mask_dangerous_morphed = cv2.morphologyEx(
        mask_dangerous_uint8, cv2.MORPH_OPEN, MORPH_KERNEL
    )
    mask_dangerous_morphed = cv2.morphologyEx(
        mask_dangerous_morphed, cv2.MORPH_CLOSE, MORPH_KERNEL
    )

    # Find contours of dangerous regions
    contours, _ = cv2.findContours(
        mask_dangerous_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    display_frame = frame2_resized.copy()
    dangerous_centers = []
    # Draw bounding boxes around dangerous regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 50:  # Filter small regions
            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                DANGER_RECT_COLOR,
                DANGER_RECT_THICKNESS,
            )
            dangerous_centers.append((x + w // 2, y + h // 2))

    # Identify grid cells containing dangerous regions
    is_cell_generally_dangerous = np.zeros((grid_rows, grid_cols), dtype=bool)
    current_convergence_cell_centers = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            cy1 = r * cell_height
            cy2 = min(new_height, cy1 + cell_height)
            cx1 = c * cell_width
            cx2 = min(new_width, cx1 + cell_width)

            # Check if any dangerous region center is within this cell
            for center_x, center_y in dangerous_centers:
                if cx1 <= center_x < cx2 and cy1 <= center_y < cy2:
                    is_cell_generally_dangerous[r, c] = True
                    break

            # Check for convergence within the cell
            if cy1 < cy2 and cx1 < cx2:
                cell_divergence = divergence[cy1:cy2, cx1:cx2]
                if (
                    cell_divergence.size > 0
                    and np.min(cell_divergence) < CONVERGENCE_THRESHOLD
                ):
                    current_convergence_cell_centers.append(
                        (cx1 + cell_width // 2, cy1 + cell_height // 2)
                    )

    # Predict future positions for convergence points
    new_predictions = []
    for center_x, center_y in current_convergence_cell_centers:
        # Get flow vector at the center
        flow_at_center_u = temporally_smoothed_flow[int(center_y), int(center_x), 0]
        flow_at_center_v = temporally_smoothed_flow[int(center_y), int(center_x), 1]

        # Predict next position
        predicted_center_x = center_x + flow_at_center_u * PREDICTION_STEPS
        predicted_center_y = center_y + flow_at_center_v * PREDICTION_STEPS

        # Clip predictions to frame boundaries
        predicted_center_x = np.clip(predicted_center_x, 0, new_width - 1)
        predicted_center_y = np.clip(predicted_center_y, 0, new_height - 1)

        new_predictions.append(
            (int(predicted_center_x), int(predicted_center_y), PREDICTION_FADE_FRAMES)
        )

    # Add new predictions and update existing ones (fade)
    active_predictions.extend(new_predictions)

    updated_predictions = []
    for pred_x, pred_y, age in active_predictions:
        new_age = age - 1
        if new_age > 0:
            updated_predictions.append((pred_x, pred_y, new_age))
    active_predictions[:] = updated_predictions  # Update the original list in place

    # Draw the grid
    for i in range(1, grid_cols):
        x = i * cell_width
        cv2.line(display_frame, (x, 0), (x, new_height), GRID_COLOR, GRID_THICKNESS)
    # Draw rightmost vertical line
    cv2.line(
        display_frame,
        (new_width - 1, 0),
        (new_width - 1, new_height),
        GRID_COLOR,
        GRID_THICKNESS,
    )

    for i in range(1, grid_rows):
        y = i * cell_height
        cv2.line(display_frame, (0, y), (new_width, y), GRID_COLOR, GRID_THICKNESS)
    # Draw bottom horizontal line
    cv2.line(
        display_frame,
        (0, new_height - 1),
        (new_width, new_height - 1),
        GRID_COLOR,
        GRID_THICKNESS,
    )

    # Highlight generally dangerous cells
    for row in range(grid_rows):
        for col in range(grid_cols):
            if is_cell_generally_dangerous[row, col]:
                cx1 = col * cell_width
                cy1 = row * cell_height
                cx2 = min(new_width, cx1 + cell_width)
                cy2 = min(new_height, cy1 + cell_height)

                cv2.rectangle(
                    display_frame,
                    (cx1, cy1),
                    (cx2, cy2),
                    GENERAL_DANGER_CELL_COLOR,
                    GENERAL_DANGER_CELL_THICKNESS,
                )

    # Tint and outline cells with strong convergence
    for r in range(grid_rows):
        for c in range(grid_cols):
            cy1 = r * cell_height
            cy2 = min(new_height, cy1 + cell_height)
            cx1 = c * cell_width
            cx2 = min(new_width, cx1 + cell_width)

            if cy1 < cy2 and cx1 < cx2:
                cell_divergence = divergence[cy1:cy2, cx1:cx2]
                if (
                    cell_divergence.size > 0
                    and np.min(cell_divergence) < CONVERGENCE_THRESHOLD
                ):
                    # Apply tint
                    display_frame[cy1:cy2, cx1:cx2] = np.clip(
                        display_frame[cy1:cy2, cx1:cx2].astype(np.float32)
                        + CONVERGENCE_THRESHOLD_TINT_COLOR,
                        0,
                        255,
                    ).astype(np.uint8)

                    # Draw outline
                    cv2.rectangle(
                        display_frame,
                        (cx1, cy1),
                        (cx2, cy2),
                        CONVERGENCE_THRESHOLD_OUTLINE_COLOR,
                        CONVERGENCE_THRESHOLD_OUTLINE_THICKNESS,
                    )

    # Draw predicted points with fading effect
    overlay = display_frame.copy()
    for pred_x, pred_y, age in active_predictions:
        alpha = np.clip(age / PREDICTION_FADE_FRAMES, 0, 1)
        cv2.circle(overlay, (pred_x, pred_y), PREDICTION_RADIUS, PREDICTION_COLOR, -1)
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

    # Prepare data for quiver plot
    angle = cv2.cartToPolar(
        temporally_smoothed_flow[..., 0],
        temporally_smoothed_flow[..., 1],
        angleInDegrees=True,
    )[1]
    hsv[..., 0] = angle / 2
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = normalized_magnitude
    # rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # This line seems unused in the provided code

    h, w = new_height, new_width
    # Select points for quiver plot
    y_coords, x_coords = np.mgrid[0:h:QUIVER_STEP, 0:w:QUIVER_STEP]

    # Ensure selected coordinates are within bounds and have corresponding flow data
    if x_coords.size > 0 and y_coords.size > 0:
        # Flatten coordinates for indexing
        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()

        # Ensure indices are within bounds
        valid_indices = (x_coords_flat < w) & (y_coords_flat < h)
        x_coords_valid = x_coords_flat[valid_indices]
        y_coords_valid = y_coords_flat[valid_indices]

        if x_coords_valid.size > 0:
            # Get flow vectors for valid points
            u_vectors_valid = temporally_smoothed_flow[
                y_coords_valid, x_coords_valid, 0
            ]
            v_vectors_valid = temporally_smoothed_flow[
                y_coords_valid, x_coords_valid, 1
            ]

            # Check for NaN or infinite values in flow vectors
            valid_flow_mask = np.isfinite(u_vectors_valid) & np.isfinite(
                v_vectors_valid
            )

            x_coords_final = x_coords_valid[valid_flow_mask]
            y_coords_final = y_coords_valid[valid_flow_mask]
            u_vectors_final = u_vectors_valid[valid_flow_mask]
            v_vectors_final = v_vectors_valid[valid_flow_mask]

            if x_coords_final.size > 0:
                # Calculate magnitude and angle for coloring and scaling
                mag_selected = np.sqrt(u_vectors_final**2 + v_vectors_final**2)
                angle_selected_rad = np.arctan2(v_vectors_final, u_vectors_final)
                angle_selected_deg = (np.degrees(angle_selected_rad) + 360) % 360

                max_mag_selected = np.max(mag_selected) if mag_selected.size > 0 else 0

                # Determine quiver scale
                if max_mag_selected > 1e-6:
                    quiver_scale = max_mag_selected / MAX_ARROW_DISPLAY_LENGTH
                    if quiver_scale < 1e-6:  # Prevent division by near zero
                        quiver_scale = 1.0
                else:
                    quiver_scale = 1.0

                # Map angles to colors using HSV colormap
                normalized_angles = angle_selected_deg / 360.0
                cmap = plt.colormaps.get_cmap("hsv")
                colors = cmap(normalized_angles)

                # Update the quiver plot
                ax.cla()  # Clear previous plot
                ax.quiver(
                    x_coords_final,
                    y_coords_final,
                    u_vectors_final,
                    v_vectors_final,
                    color=colors,
                    scale=quiver_scale,
                    angles="xy",
                    scale_units="xy",
                    pivot="mid",
                )
                ax.invert_yaxis()  # Invert y-axis to match image coordinates
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.set_title(
                    "Estimated Flow Field (Quiver Plot - Direction Color, Max Length)"
                )
                ax.axis("off")  # Hide the axes
                # plt.draw()
                plt.pause(0.001)  # Pause to allow plot to update
            else:
                # Handle case with no valid flow vectors after filtering
                ax.cla()
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.axis("off")  # Hide the axes
                # plt.draw()
                plt.pause(0.001)
        else:
            # Handle case with no valid grid points after filtering
            ax.cla()
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.axis("off")  # Hide the axes
            # plt.draw()
            plt.pause(0.001)
    else:
        # Handle case with no grid points generated
        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")  # Hide the axes
        # plt.draw()
        plt.pause(0.001)

    # Convert the quiver plot to a NumPy array using Pillow
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    quiver_img = np.array(img)

    return True, next_gray, display_frame, quiver_img
