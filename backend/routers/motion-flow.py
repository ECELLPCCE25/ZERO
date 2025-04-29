import cv2
import numpy as np
import time


def draw_flow_vectors(img, flow, step=16):
    """
    Draws optical flow vectors on an image.

    Args:
        img (np.ndarray): The image to draw on (should be in BGR format).
        flow (np.ndarray): The dense optical flow field (2-channel).
        step (int): The step size for drawing vectors (e.g., draw a vector every 16 pixels).

    Returns:
        np.ndarray: The image with flow vectors drawn.
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)  # Add 0.5 for correct rounding
    vis = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )  # Draw on a grayscale version for clarity
    vis = cv2.cvtColor(
        vis, cv2.COLOR_GRAY2BGR
    )  # Convert back to BGR to draw color lines
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_flow_hsv(flow):
    """
    Visualizes the optical flow in HSV color space.

    Args:
        flow (np.ndarray): The dense optical flow field (2-channel).

    Returns:
        np.ndarray: The optical flow visualized in BGR color space.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Set saturation to maximum

    # Angle determines the hue (color)
    hsv[..., 0] = ang * 180 / np.pi / 2

    # Magnitude determines the value (brightness)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def run_dense_optical_flow():
    """
    Captures video from webcam, computes and visualizes dense optical flow.
    """
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes (optional, used in some examples)
    # mask = np.zeros_like(old_frame)

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Start timer for FPS calculation
        start_time = time.time()

        # Calculate dense optical flow using Farneback method
        # Parameters are based on common examples and the video transcript
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Update the previous frame
        old_gray = frame_gray.copy()

        # End timer and calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # --- Visualization ---

        # Visualize flow as vectors
        flow_vectors_img = draw_flow_vectors(frame.copy(), flow)

        # Visualize flow in HSV color space
        flow_hsv_img = draw_flow_hsv(flow)

        # Display FPS on the vector visualization
        cv2.putText(
            flow_vectors_img,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the results
        cv2.imshow("Optical Flow (Vectors)", flow_vectors_img)
        cv2.imshow("Optical Flow (HSV)", flow_hsv_img)
        cv2.imshow("Original Frame", frame)

        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_dense_optical_flow()
