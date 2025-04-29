import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import sys

router = APIRouter()

DOWNSAMPLE_FACTOR = 4


def calculate_and_visualize_flow(old_gray, frame_gray):
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((frame_gray.shape[0], frame_gray.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_hsv_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_hsv_img


async def generate_optical_flow_stream(video_source: int = 0):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from: {video_source}")
        cap.release()
        return

    original_height, original_width = frame.shape[:2]
    intermediate_width = original_width // DOWNSAMPLE_FACTOR
    intermediate_height = original_height // DOWNSAMPLE_FACTOR
    old_gray = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        (intermediate_width, intermediate_height),
        interpolation=cv2.INTER_NEAREST,
    )

    print("Starting optical flow stream generator...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video source: {video_source} or error reading frame.")
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            downsampled_img = cv2.resize(
                frame_gray,
                (intermediate_width, intermediate_height),
                interpolation=cv2.INTER_NEAREST,
            )

            flow_hsv_img = calculate_and_visualize_flow(old_gray, downsampled_img)

            old_gray = downsampled_img.copy()

            pixelated_img = cv2.resize(
                flow_hsv_img,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            ret, buffer = cv2.imencode(".jpg", pixelated_img)
            if not ret:
                print("Error encoding frame.")
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            await asyncio.sleep(0.03)

    except asyncio.CancelledError:
        print("Stream cancelled by client or server shutdown.")
    except Exception as e:
        print(f"An error occurred during streaming: {e}")
    finally:
        print("Releasing video capture resource.")
        cap.release()
        print("Optical flow stream generator stopped.")


@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_optical_flow_stream(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )
