import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt  # Import matplotlib

# --- Smoothing Parameters ---
SPATIAL_SMOOTHING_KERNEL_SIZE = (
    3,
    3,
)  # Kernel size for Gaussian blur (odd numbers recommended)
TEMPORAL_SMOOTHING_FRAMES = 3  # Number of past frames to average for temporal smoothing

# --- Quiver Plot Parameters ---
QUIVER_STEP = 12  # Adjust this value to change the density of arrows
QUIVER_SCALE = 10 * QUIVER_STEP  # Adjust this to control arrow length

# Open the default camera (usually camera 0)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame
ret, frame1 = cap.read()

if not ret:
    print("Error: Could not read initial frame.")
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create a blank HSV image for visualizing the flow (color-coded)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255  # Set saturation to maximum

# Deque to store recent flow fields for temporal smoothing
flow_history = deque(maxlen=TEMPORAL_SMOOTHING_FRAMES)

# Setup the matplotlib figure for the quiver plot
plt.figure("Optical Flow Quiver Plot")
plt.ion()  # Turn on interactive mode

print("Press 'q' to exit.")

while True:
    # Read the next frame
    ret, frame2 = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the current frame to grayscale
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # --- Apply Spatial Smoothing ---
    # We apply Gaussian blur to the u and v channels of the flow field separately
    flow_u_smooth = cv2.GaussianBlur(flow[..., 0], SPATIAL_SMOOTHING_KERNEL_SIZE, 0)
    flow_v_smooth = cv2.GaussianBlur(flow[..., 1], SPATIAL_SMOOTHING_KERNEL_SIZE, 0)
    spatially_smoothed_flow = np.stack((flow_u_smooth, flow_v_smooth), axis=-1)

    # --- Apply Temporal Smoothing ---
    # Add the current spatially smoothed flow to the history
    flow_history.append(spatially_smoothed_flow)

    # Calculate the average flow from the history
    temporally_smoothed_flow = np.mean(list(flow_history), axis=0)

    # --- Now, 'temporally_smoothed_flow' is your smoothed dense flow field ---
    # We will visualize this smoothed flow using both color-coding and quiver plot.

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
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the HSV flow visualization image to BGR for displaying
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Quiver Plot Visualization ---
    # Get the dimensions of the flow field
    h, w = temporally_smoothed_flow.shape[:2]

    # Create a grid of coordinates
    y_coords, x_coords = np.mgrid[0:h:QUIVER_STEP, 0:w:QUIVER_STEP]

    # Get the flow vectors for the selected coordinates
    u_vectors = temporally_smoothed_flow[y_coords, x_coords, 0]
    v_vectors = temporally_smoothed_flow[y_coords, x_coords, 1]

    # Clear the previous plot
    plt.clf()

    # Create the quiver plot
    plt.quiver(
        x_coords, y_coords, u_vectors, v_vectors, color="black", scale=QUIVER_SCALE
    )  # Use the defined scale

    # Invert the y-axis to match image coordinates
    plt.gca().invert_xaxis

    # Set equal aspect ratio
    plt.gca().set_aspect("equal", adjustable="box")

    # Set plot limits (optional, can help with consistent view)
    plt.xlim(0, w)
    plt.ylim(h, 0)

    plt.title("Estimated Flow Field (Quiver Plot)")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")

    # Draw the plot and pause briefly for update
    plt.draw()
    plt.pause(0.001)  # Use a smaller pause for faster updates

    # Display the original webcam feed and the smoothed color-coded flow visualization
    cv2.imshow("Webcam Feed", frame2)
    cv2.imshow("Optical Flow (Smoothed Color)", rgb_flow)

    # Update the previous frame for the next iteration
    prev_gray = next_gray

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the matplotlib plot window
plt.close("Optical Flow Quiver Plot")
# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
