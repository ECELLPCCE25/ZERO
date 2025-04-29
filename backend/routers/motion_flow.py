import cv2
import numpy as np
from fastapi import APIRouter, WebSocket

import base64
from fastapi.responses import StreamingResponse
import asyncio
import time

router = APIRouter()


def decode_frame(data):
    encoded_data = data.split(",")[1]
    np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    encoded_frame = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_frame}"


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


@router.get("/ipcam/{stream_id}")
async def ipcam_endpoint(stream_id: str):
    async def generate():
        video_url = "http://192.168.69.79:4747/video"
        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            print(
                f"Error: Could not open video stream from {video_url} for stream ID: {stream_id}"
            )
            return

        print(f"Video stream opened from {video_url} for stream ID: {stream_id}")

        old_gray = None

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame from stream {video_url}")
            cap.release()
            return

        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Stream {video_url} ended for stream ID: {stream_id}.")
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            start_time = time.time()

            flow_hsv_img = calculate_and_visualize_flow(old_gray, frame_gray)

            old_gray = frame_gray.copy()

            try:
                results = model(frame, verbose=False)
                person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)
                annotated_frame = results[0].plot()

                cv2.putText(
                    annotated_frame,
                    f"People Count: {person_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            except NameError:
                print("YOLO model is not loaded. Skipping inference.")
                annotated_frame = frame.copy()
                person_count = "N/A"
                cv2.putText(
                    annotated_frame,
                    "YOLO model not loaded",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            h_annotated, w_annotated = annotated_frame.shape[:2]
            h_flow, w_flow = flow_hsv_img.shape[:2]

            if h_annotated != h_flow:
                flow_hsv_img = cv2.resize(
                    flow_hsv_img, (int(w_flow * h_annotated / h_flow), h_annotated)
                )
                h_flow, w_flow = flow_hsv_img.shape[:2]

            combined_frame = np.hstack((annotated_frame, flow_hsv_img))

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            print(f"Stream {stream_id} - FPS: {fps:.2f}, People Count: {person_count}")

            _, buffer = cv2.imencode(".jpg", combined_frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        cap.release()
        print(f"Video stream released for stream ID: {stream_id}")

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace;boundary=frame"
    )
