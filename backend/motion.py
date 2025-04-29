import cv2
import numpy as np
import time


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


def process_single_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    old_gray = None

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from: {video_path}")
        cap.release()
        return

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video: {video_path}")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        flow_hsv_img = calculate_and_visualize_flow(old_gray, frame_gray)
        old_gray = frame_gray.copy()

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

        cv2.imshow("Original and Flow", flow_hsv_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "dataset/1.mp4"  # Replace with the actual path to your video file
    process_single_video(video_file)
