import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors  # Import mcolors for color handling

# --- Smoothing Parameters ---
SPATIAL_SMOOTHING_KERNEL_SIZE = (
    3,
    3,
)  # Kernel size for Gaussian blur (odd numbers recommended)
TEMPORAL_SMOOTHING_FRAMES = 3  # Number of past frames to average for temporal smoothing

# --- Quiver Plot Parameters ---
QUIVER_STEP = 8  # Adjust this value to change the density of arrows
MAX_ARROW_DISPLAY_LENGTH = (
    20.0  # Desired max length of the longest arrow in plot units (relative to axes)
)

# --- Analysis & Visualization Parameters ---
# Threshold for detecting significant convergence (negative divergence)
CONVERGENCE_THRESHOLD = -50.0  # Example threshold, needs tuning!
# Threshold for detecting significant divergence (positive divergence)
DIVERGENCE_THRESHOLD = 50.0  # Example threshold, needs tuning!
# Threshold for detecting significant curl (swirling)
CURL_THRESHOLD = 30.0  # Example threshold, needs tuning!

# Parameters for drawing detected points
DRAW_RADIUS = 5
DRAW_COLOR_CONVERGENCE = (0, 255, 0)  # Green for convergence (going towards)
DRAW_COLOR_DIVERGENCE = (0, 0, 255)  # Red for divergence (spreading out)
DRAW_COLOR_CURL = (255, 0, 0)  # Blue for curl (swirling)
DRAW_THICKNESS = -1  # -1 for filled circle

# --- Frame Resizing Parameter ---
MAX_WIDTH = 420  # Maximum width for displayed frames

# Open the video source
# cap = cv2.VideoCapture(0) # Use camera
cap = cv2.VideoCapture("dataset/1.mp4")  # Use video file

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
    # Flow will have the dimensions of the resized frames (new_height, new_width)
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

    # Calculate gradients using np.gradient
    # np.gradient returns gradients w.r.t rows (y) and columns (x)
    du_dy, du_dx = np.gradient(flow_u)
    dv_dy, dv_dx = np.gradient(flow_v)

    # Calculate Divergence: du/dx + dv/dy
    divergence = du_dx + dv_dy

    # Calculate Curl (2D magnitude): dv/dx - du/dy
    curl = dv_dx - du_dy

    # --- Identify Significant Regions ---
    # Find pixels where divergence is below the negative threshold (convergence)
    convergence_points = np.argwhere(divergence < CONVERGENCE_THRESHOLD)

    # Find pixels where divergence is above the positive threshold (divergence)
    divergence_points = np.argwhere(divergence > DIVERGENCE_THRESHOLD)

    # Find pixels where absolute curl is above the threshold (swirling)
    curl_points = np.argwhere(np.abs(curl) > CURL_THRESHOLD)

    # --- Visualization ---

    # --- Color-Coded Flow Visualization ---
    # Compute the magnitude and angle of the smoothed flow vectors
    magnitude, angle = cv2.cartToPolar(
        temporally_smoothed_flow[..., 0],
        temporally_smoothed_flow[..., 1],
        angleInDegrees=True,
    )
    # Update the hue channel based on the angle (direction)
    hsv[..., 0] = angle / 2
    # Update the value channel based on the magnitude (speed)
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = normalized_magnitude
    # Convert the HSV flow visualization image to BGR for displaying
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Draw Detected Points on Frame ---
    # Create a copy of the resized frame to draw on
    display_frame = frame2_resized.copy()

    # Draw convergence points (where crowd is heading)
    for y, x in convergence_points:
        cv2.circle(
            display_frame, (x, y), DRAW_RADIUS, DRAW_COLOR_CONVERGENCE, DRAW_THICKNESS
        )

    # Draw divergence points (where crowd is spreading from)
    for y, x in divergence_points:
        cv2.circle(
            display_frame, (x, y), DRAW_RADIUS, DRAW_COLOR_DIVERGENCE, DRAW_THICKNESS
        )

    # Draw curl points (swirling intersections)
    for y, x in curl_points:
        cv2.circle(display_frame, (x, y), DRAW_RADIUS, DRAW_COLOR_CURL, DRAW_THICKNESS)

    # --- Quiver Plot Visualization ---
    # Use the new dimensions for the grid
    h, w = new_height, new_width
    # Create a grid of coordinates based on QUIVER_STEP
    y_coords, x_coords = np.mgrid[0:h:QUIVER_STEP, 0:w:QUIVER_STEP]

    # Check if any points are selected for the quiver plot
    if x_coords.size == 0 or y_coords.size == 0:
        print("No points selected for quiver plot. Skipping quiver.")
        # Clear axes and set limits/title even if skipping quiver
        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title("Estimated Flow Field (Quiver Plot - Direction Color, Max Length)")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        plt.draw()
        plt.pause(0.001)
    else:
        # Get flow vectors for selected coordinates
        u_vectors = temporally_smoothed_flow[y_coords, x_coords, 0]
        v_vectors = temporally_smoothed_flow[y_coords, x_coords, 1]

        # --- Handle potential NaN/Inf values in vectors ---
        # Create a mask for valid (non-NaN/Inf) vector components
        valid_mask = np.isfinite(u_vectors) & np.isfinite(v_vectors)

        # Filter out invalid vectors and their corresponding coordinates
        x_coords_valid = x_coords[valid_mask]
        y_coords_valid = y_coords[valid_mask]
        u_vectors_valid = u_vectors[valid_mask]
        v_vectors_valid = v_vectors[valid_mask]

        # Only proceed if there are valid vectors to plot
        if x_coords_valid.size > 0:
            # Calculate magnitude and angle for the selected valid vectors
            mag_selected = np.sqrt(u_vectors_valid**2 + v_vectors_valid**2)
            angle_selected_rad = np.arctan2(
                v_vectors_valid, u_vectors_valid
            )  # Get angle in radians
            angle_selected_deg = np.degrees(angle_selected_rad)  # Convert to degrees
            angle_selected_deg = (angle_selected_deg + 360) % 360  # Ensure 0-360 range

            # --- Set Quiver Scale for Max Length ---
            # Determine the maximum magnitude of the *valid* vectors being plotted
            max_mag_selected = np.max(mag_selected) if mag_selected.size > 0 else 0

            if max_mag_selected > 1e-6:  # Avoid division by zero or near-zero
                # Calculate scale such that an arrow with max_mag_selected has a display length of MAX_ARROW_DISPLAY_LENGTH
                # If scale_units='xy', arrow length in data units is `magnitude`. Displayed length is `magnitude / scale`.
                # We want MAX_ARROW_DISPLAY_LENGTH = max_mag_selected / scale
                quiver_scale = max_mag_selected / MAX_ARROW_DISPLAY_LENGTH
                # Ensure scale is not excessively large which would make arrows disappear
                if quiver_scale < 1e-6:
                    quiver_scale = 1.0  # Prevent near-zero scale

            else:
                quiver_scale = 1.0  # Default scale if max magnitude is zero

            # --- Set Quiver Colors by Direction ---
            # Normalize angles (0-360) to (0-1) for colormap
            normalized_angles = angle_selected_deg / 360.0
            # Use the updated way to get the colormap
            cmap = plt.colormaps.get_cmap("hsv")
            # Get colors from 'hsv' colormap. This returns an (N, 4) array of RGBA values.
            colors = cmap(normalized_angles)

            # --- Draw Quiver Plot ---
            ax.cla()  # Clear current axes

            ax.quiver(
                x_coords_valid,
                y_coords_valid,
                u_vectors_valid,
                v_vectors_valid,
                color=colors,  # Use the generated colors array (N, 4)
                scale=quiver_scale,  # Use the calculated scale
                angles="xy",  # Interpret U,V as vector components in data coordinates
                scale_units="xy",  # Units for the scale parameter (data units)
                pivot="mid",  # Pivot the arrow at its base (or 'mid')
            )

            # --- Set Plot Limits and Title ---
            ax.invert_yaxis()  # Invert the y-axis to match image coordinates
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_title(
                "Estimated Flow Field (Quiver Plot - Direction Color, Max Length)"
            )
            ax.set_xlabel("X-coordinate")
            ax.set_ylabel("Y-coordinate")

            # Draw the plot and pause briefly for update
            plt.draw()
            plt.pause(0.001)
        else:
            print("No valid vectors to plot for quiver.")
            # Clear axes and set limits/title even if no valid vectors
            ax.cla()
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

    # --- Display Windows ---
    # Display the original webcam feed and the smoothed color-coded flow visualization
    cv2.imshow("Webcam Feed + Analysis", display_frame)
    # cv2.imshow("Optical Flow (Smoothed Color)", rgb_flow) # You can also display the color flow

    # Optional debug displays (normalize over the potentially smaller range of values in the resized flow)
    # Normalize divergence for display
    divergence_display = cv2.normalize(
        divergence, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    cv2.imshow("Divergence Map (Green=Conv, Red=Div)", divergence_display)

    # Display the curl map (optional)
    # Normalize curl for display
    curl_display = cv2.normalize(np.abs(curl), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("Absolute Curl Map (Brightness=Swirling)", curl_display)

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
