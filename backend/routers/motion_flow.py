import cv2
import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import time

router = APIRouter()

# Configuration parameters - adjust as needed
DOWNSAMPLE_FACTOR = 11  # Back to original value
MOTION_THRESHOLD = 30  # Lower threshold for visibility
FRAME_SKIP = 0  # No frame skipping initially


def calculate_and_visualize_flow(old_gray, frame_gray, flow=None, hsv=None):
    """
    Calculate and visualize optical flow with optimizations for speed and sensitivity.
    """
    # Reuse flow array for better performance
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, frame_gray, flow, 0.5, 3, 12, 3, 5, 1.2, 0
    )

    # Calculate magnitude and angle of flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Initialize hsv array if not already done
    if hsv is None:
        hsv = np.zeros((frame_gray.shape[0], frame_gray.shape[1], 3), dtype=np.uint8)

    # Apply a softer threshold that doesn't completely zero out small movements
    # This ensures we still see something while reducing noise
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255

    # Apply threshold to magnitude but with a smooth falloff
    # This ensures there's always some visible output
    mag_thresholded = np.maximum(0, mag - MOTION_THRESHOLD)
    hsv[..., 2] = cv2.normalize(mag_thresholded, None, 0, 255, cv2.NORM_MINMAX)

    # If everything is below threshold, show at least something
    if np.max(hsv[..., 2]) < 10:  # Check if almost black
        # Show small movements at lower intensity
        hsv[..., 2] = cv2.normalize(mag, None, 0, 100, cv2.NORM_MINMAX)

    # Convert to BGR
    flow_hsv_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_hsv_img, flow, hsv


async def generate_optical_flow_stream(device: str = "0"):
    """
    Generate an optimized optical flow stream.
    """
    video_url = f"http://{device}:4747/video"
    cap = cv2.VideoCapture(video_url)

    # Set buffer size to 1 to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_url}")
        return

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from: {video_url}")
        cap.release()
        return

    # Calculate dimensions once
    original_height, original_width = frame.shape[:2]
    intermediate_width = original_width // DOWNSAMPLE_FACTOR
    intermediate_height = original_height // DOWNSAMPLE_FACTOR

    # Pre-allocate arrays for better performance
    old_gray = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        (intermediate_width, intermediate_height),
        interpolation=cv2.INTER_NEAREST,
    )
    downsampled_img = np.zeros(
        (intermediate_height, intermediate_width), dtype=np.uint8
    )
    flow = None
    hsv = None
    frame_count = 0

    # Create a buffer for JPEG encoding
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Higher quality for visibility

    print("Starting optical flow stream generator...")
    last_time = time.time()
    fps_time = time.time()
    frames_processed = 0

    try:
        while True:
            # Frame skipping if enabled
            frame_count += 1
            if FRAME_SKIP > 0 and frame_count % (FRAME_SKIP + 1) != 0:
                # Still need to read the frame to advance the buffer
                ret = cap.grab()
                if not ret:
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                print(f"End of video source: {video_url} or error reading frame.")
                break

            # Resize directly to destination array for better performance
            cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (intermediate_width, intermediate_height),
                dst=downsampled_img,
                interpolation=cv2.INTER_NEAREST,
            )

            # Calculate and visualize optical flow, reusing arrays
            flow_hsv_img, flow, hsv = calculate_and_visualize_flow(
                old_gray, downsampled_img, flow, hsv
            )

            # Debug - print mean magnitude to help diagnose black screen
            if frame_count % 30 == 0:  # Only print occasionally
                if flow is not None:
                    mean_flow = np.mean(np.abs(flow))
                    print(f"Mean flow magnitude: {mean_flow:.4f}")

            # Use direct array copying for better performance
            np.copyto(old_gray, downsampled_img)

            # Resize to original dimensions
            pixelated_img = cv2.resize(
                flow_hsv_img,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Encode with optimized parameters
            ret, buffer = cv2.imencode(".jpg", pixelated_img, encode_params)
            if not ret:
                print("Error encoding frame.")
                continue

            frame_bytes = buffer.tobytes()

            # Yield the frame
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            # Track FPS
            current_time = time.time()  # Assign current_time first
            if current_time - fps_time > 5.0:
                fps = frames_processed / (current_time - fps_time)
                print(f"Processing FPS: {fps:.2f}")
                frames_processed = 0
                fps_time = current_time

            # Adaptive sleep time based on processing performance
            processing_time = time.time() - last_time
            sleep_time = max(0.01, 0.03 - processing_time)  # At least 10ms sleep
            await asyncio.sleep(sleep_time)
            last_time = time.time()

    except asyncio.CancelledError:
        print("Stream cancelled by client or server shutdown.")
    except Exception as e:
        print(f"An error occurred during streaming: {e}")
    finally:
        print("Releasing video capture resource.")
        cap.release()
        print("Optical flow stream generator stopped.")


@router.get("/video_feed")
async def video_feed(device: str = Query(...), id: str = Query(...)):
    """
    Stream endpoint that returns the optical flow visualization.
    """
    return StreamingResponse(
        generate_optical_flow_stream(device),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )
