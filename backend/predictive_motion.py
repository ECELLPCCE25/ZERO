import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math  # Import math for sqrt and round

# --- Smoothing Parameters ---
SPATIAL_SMOOTHING_KERNEL_SIZE = (
    5,
    5,
)  # Kernel size for Gaussian blur (increase slightly for better region detection)
TEMPORAL_SMOOTHING_FRAMES = (
    2  # Number of past frames to average for temporal smoothing (increase slightly)
)

# --- Quiver Plot Parameters ---
QUIVER_STEP = (
    16  # Adjust this value to change the density of arrows (increased step for clarity)
)
MAX_ARROW_DISPLAY_LENGTH = (
    50.0  # Desired max length of the longest arrow in plot units (relative to axes)
)

# --- Analysis & Visualization Parameters ---
# Threshold for detecting significant convergence (negative divergence)
CONVERGENCE_THRESHOLD = (
    -0.6  # Highlight cells where min divergence is BELOW this (more negative)
)
# Threshold for detecting significant divergence (positive divergence)
DIVERGENCE_THRESHOLD = (
    0.6  # Example threshold, needs tuning! (Used for general danger region detection)
)
# Threshold for detecting significant curl (swirling)
CURL_THRESHOLD = (
    0.2  # Example threshold, needs tuning! (Used for general danger region detection)
)
# Threshold for significant movement magnitude
MAGNITUDE_THRESHOLD = 0.8  # Example threshold, tune this based on typical flow speeds

# Parameters for drawing danger regions (rectangles)
DANGER_RECT_COLOR = (0, 80, 255)  # Orange/Red color for danger rectangles
DANGER_RECT_THICKNESS = 2  # Thickness of the danger rectangle border

# Parameters for the Grid
NUM_GRID_CELLS = (
    15  # Desired approximate total number of grid cells (e.g., 9 for roughly 3x3)
)
GRID_COLOR = (200, 200, 200)  # Light grey for grid lines
GRID_THICKNESS = 1  # Thickness of grid lines

# Parameters for highlighting generally dangerous grid cells (based on Divergence OR Curl OR Magnitude thresholds)
GENERAL_DANGER_CELL_COLOR = (
    0,
    0,
    255,
)  # Red color for highlighting generally dangerous cells
GENERAL_DANGER_CELL_THICKNESS = (
    2  # Thickness of the highlight border (use positive for border)
)

# Parameters for highlighting cells that meet the convergence threshold
CONVERGENCE_THRESHOLD_TINT_COLOR = (
    100,
    0,
    0,
)  # BGR color to add for blue tint (e.g., 100 Blue, 0 Green, 0 Red)
CONVERGENCE_THRESHOLD_OUTLINE_COLOR = (255, 255, 255)  # White color for the outline
CONVERGENCE_THRESHOLD_OUTLINE_THICKNESS = 1  # Thin thickness for the outline

# --- Prediction Parameters ---
PREDICTION_STEPS = (
    1  # Number of frames to predict into the future (e.g., 1 for next frame)
)
PREDICTION_COLOR = (0, 165, 255)  # Standard Orange color for predicted locations (BGR)
PREDICTION_RADIUS = 5  # Radius of the circle marker for prediction
PREDICTION_FADE_FRAMES = 30  # Number of frames for the prediction marker to fade out

# --- Frame Resizing Parameter ---
MAX_WIDTH = 640  # Maximum width for displayed frames (Increased for better detail)

# --- Morphological Operations for Region Detection ---
# Kernel for morphological operations (connecting nearby dangerous pixels)
MORPH_KERNEL = np.ones((5, 5), np.uint8)  # Example kernel size


# Open the video source
# cap = cv2.VideoCapture(0)  # Use camera
cap = cv2.VideoCapture("dataset/5.mp4")  # Use video file

# Check if the video source opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Get original frame dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Calculate new dimensions if max_width is exceeded, maintaining aspect ratio
if original_width > MAX_WIDTH:
    new_width = MAX_WIDTH
    new_height = int(original_height * (MAX_WIDTH / original_width))
else:
    new_width = original_width
    new_height = original_height

print(f"Original dimensions: {original_width}x{original_height}")
print(f"Processing dimensions: {new_width}x{new_height}")

# Read the first frame
ret, frame1 = cap.read()

if not ret:
    print("Error: Could not read initial frame.")
    exit()

# Resize the first frame
frame1_resized = cv2.resize(frame1, (new_width, new_height))

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)

# Create a blank HSV image for visualizing the flow (color-coded), based on new dimensions
hsv = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # Specify dtype
hsv[..., 1] = 255  # Set saturation to maximum

# Deque to store recent flow fields for temporal smoothing
flow_history = deque(maxlen=TEMPORAL_SMOOTHING_FRAMES)

# List to store active predictions (each is a tuple: (x, y, age))
active_predictions = []

# --- Calculate Grid Dimensions based on NUM_GRID_CELLS and Aspect Ratio ---
# Calculate the aspect ratio of the resized frame
aspect_ratio = new_width / new_height

# Estimate the number of columns needed for square-like cells
# Total cells = rows * cols
# cols / rows = aspect_ratio => cols = rows * aspect_ratio
# Total cells = rows * (rows * aspect_ratio) = rows^2 * aspect_ratio
# rows^2 = Total cells / aspect_ratio
estimated_rows = round(math.sqrt(NUM_GRID_CELLS / aspect_ratio))
estimated_cols = round(estimated_rows * aspect_ratio)

# Ensure we have at least 1 row and 1 column
grid_rows = max(1, estimated_rows)
grid_cols = max(1, estimated_cols)

# Adjust slightly to get closer to the desired number of cells if possible
# Prioritize getting closer to NUM_GRID_CELLS while maintaining aspect ratio
current_cells = grid_rows * grid_cols
if current_cells < NUM_GRID_CELLS:
    # Try increasing rows or cols based on which would make cells more square
    if aspect_ratio > 1:  # Wider than tall, increasing cols might be better
        grid_cols = max(1, grid_cols + 1)
    else:  # Taller than wide, increasing rows might be better
        grid_rows = max(1, grid_rows + 1)

# Recalculate cell dimensions based on the determined grid size
cell_width = new_width // grid_cols
cell_height = new_height // grid_rows

# Ensure cell dimensions are at least 1 pixel
cell_width = max(1, cell_width)
cell_height = max(1, cell_height)

print(f"Calculated grid size: {grid_rows}x{grid_cols} ({grid_rows * grid_cols} cells)")
print(f"Cell dimensions: {cell_width}x{cell_height}")


# Setup the matplotlib figure for the quiver plot
plt.figure("Optical Flow Quiver Plot")
plt.ion()  # Turn on interactive mode
ax = plt.gca()  # Get the axes object

print("Press 'q' to exit.")

while True:
    # Read the next frame
    ret, frame2 = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        # Loop video if reading from file and not a camera
        if isinstance(cap, cv2.VideoCapture) and cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
            print("Looping video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Set frame position to the beginning
            ret, frame2 = cap.read()  # Read the first frame again
            if not ret:  # Still can't read? Something is wrong
                print("Error looping video.")
                break
        else:  # If it's a camera or cannot loop
            break

    # Resize the current frame
    frame2_resized = cv2.resize(frame2, (new_width, new_height))

    # Convert the current frame to grayscale
    next_gray = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # --- Apply Spatial Smoothing ---
    flow_u_smooth = cv2.GaussianBlur(flow[..., 0], SPATIAL_SMOOTHING_KERNEL_SIZE, 0)
    flow_v_smooth = cv2.GaussianBlur(flow[..., 1], SPATIAL_SMOOTHING_KERNEL_SIZE, 0)
    spatially_smoothed_flow = np.stack((flow_u_smooth, flow_v_smooth), axis=-1)

    # --- Apply Temporal Smoothing ---
    flow_history.append(spatially_smoothed_flow)
    temporally_smoothed_flow = np.mean(list(flow_history), axis=0)

    # --- Analyze Smoothed Flow for Divergence and Curl ---
    flow_u = temporally_smoothed_flow[..., 0]
    flow_v = temporally_smoothed_flow[..., 1]

    du_dy, du_dx = np.gradient(flow_u)
    dv_dy, dv_dx = np.gradient(flow_v)

    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy

    # --- Calculate Magnitude ---
    magnitude, _ = cv2.cartToPolar(flow_u, flow_v)

    # --- Danger Region Detection (using Divergence, Curl, AND Magnitude) ---
    # Create binary masks based on thresholds
    mask_divergence = divergence > DIVERGENCE_THRESHOLD
    mask_curl = np.abs(curl) > CURL_THRESHOLD
    mask_magnitude = (
        magnitude > MAGNITUDE_THRESHOLD
    )  # New mask for significant movement

    # Combine masks: A pixel is considered potentially dangerous if it has
    # significant divergence OR significant curl AND significant magnitude.
    # This gives more weight to areas where the dangerous flow patterns are strong AND fast.
    mask_dangerous = (mask_divergence | mask_curl) & mask_magnitude

    # Optional: Apply morphological operations to connect nearby dangerous pixels
    mask_dangerous_uint8 = mask_dangerous.astype(np.uint8) * 255

    mask_dangerous_morphed = cv2.morphologyEx(
        mask_dangerous_uint8, cv2.MORPH_OPEN, MORPH_KERNEL
    )
    mask_dangerous_morphed = cv2.morphologyEx(
        mask_dangerous_morphed, cv2.MORPH_CLOSE, MORPH_KERNEL
    )

    # Find contours on the morphed danger mask
    contours, _ = cv2.findContours(
        mask_dangerous_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- Visualization ---

    display_frame = frame2_resized.copy()

    # --- Draw Danger Rectangles ---
    dangerous_centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 50:  # Minimum area threshold for a valid danger region
            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                DANGER_RECT_COLOR,
                DANGER_RECT_THICKNESS,
            )
            dangerous_centers.append((x + w // 2, y + h // 2))

    # --- Draw Grid and Highlight Dangerous Cells ---
    # Recalculate cell dimensions in case frame size changes (though fixed here)
    cell_width = new_width // grid_cols
    cell_height = new_height // grid_rows
    cell_width = max(1, cell_width)  # Ensure at least 1 pixel
    cell_height = max(1, cell_height)  # Ensure at least 1 pixel

    is_cell_generally_dangerous = np.zeros((grid_rows, grid_cols), dtype=bool)
    current_convergence_cell_centers = []  # Store centers of convergence cells for this frame

    for r in range(grid_rows):
        for c in range(grid_cols):
            cy1 = r * cell_height
            cy2 = min(new_height, cy1 + cell_height)  # Clamp bounds
            cx1 = c * cell_width
            cx2 = min(new_width, cx1 + cell_width)  # Clamp bounds

            # Check if any dangerous region centers fall into this cell
            for center_x, center_y in dangerous_centers:
                if cx1 <= center_x < cx2 and cy1 <= center_y < cy2:
                    is_cell_generally_dangerous[r, c] = True
                    break  # No need to check other centers for this cell

            # Check if the minimum divergence in this cell meets the convergence threshold
            if cy1 < cy2 and cx1 < cx2:  # Ensure the slice is valid
                cell_divergence = divergence[cy1:cy2, cx1:cx2]
                if cell_divergence.size > 0:
                    # If the minimum divergence is below the threshold (more negative)
                    if np.min(cell_divergence) < CONVERGENCE_THRESHOLD:
                        # Store the center of the convergence cell for prediction
                        current_convergence_cell_centers.append(
                            (cx1 + cell_width // 2, cy1 + cell_height // 2)
                        )

    # --- Generate New Predictions ---
    new_predictions = []
    for center_x, center_y in current_convergence_cell_centers:
        # Get the flow vector at the center of the convergence cell
        # Note: This is a simplification.
        flow_at_center_u = temporally_smoothed_flow[int(center_y), int(center_x), 0]
        flow_at_center_v = temporally_smoothed_flow[int(center_y), int(center_x), 1]

        # Predict the future center position
        predicted_center_x = center_x + flow_at_center_u * PREDICTION_STEPS
        predicted_center_y = center_y + flow_at_center_v * PREDICTION_STEPS

        # Clamp predicted positions to stay within frame bounds
        predicted_center_x = np.clip(predicted_center_x, 0, new_width - 1)
        predicted_center_y = np.clip(predicted_center_y, 0, new_height - 1)

        # Add the new prediction with full age
        new_predictions.append(
            (int(predicted_center_x), int(predicted_center_y), PREDICTION_FADE_FRAMES)
        )

    # Add new predictions to the active list
    active_predictions.extend(new_predictions)

    # --- Update and Filter Active Predictions (for fading) ---
    # Decrement age and keep only predictions with age > 0
    updated_predictions = []
    for pred_x, pred_y, age in active_predictions:
        new_age = age - 1
        if new_age > 0:
            updated_predictions.append((pred_x, pred_y, new_age))
    active_predictions = updated_predictions

    # Draw grid lines
    for i in range(1, grid_cols):
        x = i * cell_width
        cv2.line(display_frame, (x, 0), (x, new_height), GRID_COLOR, GRID_THICKNESS)
    # Draw the rightmost vertical line explicitly if it doesn't align perfectly
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
    # Draw the bottom horizontal line explicitly if it doesn't align perfectly
    cv2.line(
        display_frame,
        (0, new_height - 1),
        (new_width, new_height - 1),
        GRID_COLOR,
        GRID_THICKNESS,
    )

    # Highlight generally dangerous cells (red border)
    for row in range(grid_rows):
        for col in range(grid_cols):
            if is_cell_generally_dangerous[row, col]:
                cx1 = col * cell_width
                cy1 = row * cell_height
                cx2 = min(new_width, cx1 + cell_width)  # Clamp bounds
                cy2 = min(new_height, cy1 + cell_height)  # Clamp bounds

                cv2.rectangle(
                    display_frame,
                    (cx1, cy1),
                    (cx2, cy2),
                    GENERAL_DANGER_CELL_COLOR,
                    GENERAL_DANGER_CELL_THICKNESS,
                )

    # --- Highlight Cells Meeting Convergence Threshold (Tint and Outline) ---
    # We will highlight the *current* convergence cells before drawing predictions
    for r in range(grid_rows):
        for c in range(grid_cols):
            cy1 = r * cell_height
            cy2 = min(new_height, cy1 + cell_height)  # Clamp bounds
            cx1 = c * cell_width
            cx2 = min(new_width, cx1 + cell_width)  # Clamp bounds

            if cy1 < cy2 and cx1 < cx2:  # Ensure slice is valid
                cell_divergence = divergence[cy1:cy2, cx1:cx2]
                if (
                    cell_divergence.size > 0
                    and np.min(cell_divergence) < CONVERGENCE_THRESHOLD
                ):
                    # Apply blue tint to the cell area
                    display_frame[cy1:cy2, cx1:cx2] = np.clip(
                        display_frame[cy1:cy2, cx1:cx2].astype(np.float32)
                        + CONVERGENCE_THRESHOLD_TINT_COLOR,
                        0,
                        255,
                    ).astype(np.uint8)

                    # Draw a thin white outline around the cell
                    cv2.rectangle(
                        display_frame,
                        (cx1, cy1),
                        (cx2, cy2),
                        CONVERGENCE_THRESHOLD_OUTLINE_COLOR,
                        CONVERGENCE_THRESHOLD_OUTLINE_THICKNESS,
                    )

    # --- Draw Fading Predictions (Circles) ---
    # Create an overlay for drawing semi-transparent circles
    overlay = display_frame.copy()

    for pred_x, pred_y, age in active_predictions:
        # Calculate alpha based on age (linear fade)
        alpha = age / PREDICTION_FADE_FRAMES
        # Ensure alpha is between 0 and 1
        alpha = np.clip(alpha, 0, 1)

        # Draw the filled circle on the overlay
        cv2.circle(
            overlay,
            (pred_x, pred_y),
            PREDICTION_RADIUS,
            PREDICTION_COLOR,
            -1,  # Filled circle
        )

        # Blend the overlay with the main frame
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

    # --- Color-Coded Flow Visualization (Optional, can comment out imshow) ---
    # Compute the magnitude and angle of the smoothed flow vectors
    angle = cv2.cartToPolar(
        temporally_smoothed_flow[..., 0],
        temporally_smoothed_flow[..., 1],
        angleInDegrees=True,
    )[1]
    hsv[..., 0] = angle / 2
    # Update the value channel based on the magnitude (speed)
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = normalized_magnitude
    # Convert the HSV flow visualization image to BGR for displaying
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Quiver Plot Visualization ---
    h, w = new_height, new_width
    y_coords, x_coords = np.mgrid[0:h:QUIVER_STEP, 0:w:QUIVER_STEP]

    if x_coords.size > 0 and y_coords.size > 0:  # Check if grid is not empty
        u_vectors = temporally_smoothed_flow[y_coords, x_coords, 0]
        v_vectors = temporally_smoothed_flow[y_coords, x_coords, 1]

        valid_mask = np.isfinite(u_vectors) & np.isfinite(v_vectors)

        x_coords_valid = x_coords[valid_mask]
        y_coords_valid = y_coords[valid_mask]
        u_vectors_valid = u_vectors[valid_mask]
        v_vectors_valid = v_vectors[valid_mask]

        if x_coords_valid.size > 0:
            mag_selected = np.sqrt(u_vectors_valid**2 + v_vectors_valid**2)
            angle_selected_rad = np.arctan2(v_vectors_valid, u_vectors_valid)
            angle_selected_deg = (np.degrees(angle_selected_rad) + 360) % 360

            max_mag_selected = np.max(mag_selected) if mag_selected.size > 0 else 0

            if max_mag_selected > 1e-6:
                quiver_scale = max_mag_selected / MAX_ARROW_DISPLAY_LENGTH
                if quiver_scale < 1e-6:
                    quiver_scale = 1.0
            else:
                quiver_scale = 1.0

            normalized_angles = angle_selected_deg / 360.0
            cmap = plt.colormaps.get_cmap("hsv")
            colors = cmap(normalized_angles)

            ax.cla()
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
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_title(
                "Estimated Flow Field (Quiver Plot - Direction Color, Max Length)"
            )
            ax.set_xlabel("X-coordinate")
            ax.set_ylabel("Y-coordinate")
            plt.draw()
            plt.pause(0.001)
        else:
            ax.cla()
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_title("Estimated Flow Field (Quiver Plot - No Valid Vectors)")
            ax.set_xlabel("X-coordinate")
            ax.set_ylabel("Y-coordinate")
            plt.draw()
            plt.pause(0.001)
    else:
        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title("Estimated Flow Field (Quiver Plot - No Grid Points)")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        plt.draw()
        plt.pause(0.001)

    # --- Display Windows ---
    # Display the main frame with rectangles, grid, highlight, and predictions
    cv2.imshow(
        "Webcam Feed + Crowd Analysis (Danger Regions, Convergence Highlight & Fading Prediction)",
        display_frame,
    )

    # Removed: cv2.imshow("Optical Flow (Smoothed Color)", rgb_flow)
    # Removed: cv2.imshow("Divergence Map (Normalized Grayscale)", divergence_display)
    # Removed: cv2.imshow("Absolute Curl Map (Brightness=Swirling)", curl_display)
    # Removed: cv2.imshow("Highest Convergence Area", convergence_section_display)

    # Update the previous frame for the next iteration (use the resized grayscale frame)
    prev_gray = next_gray

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the matplotlib plot window
plt.close("Optical Flow Quiver Plot")
# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
