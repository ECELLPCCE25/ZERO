import cv2
import base64
import asyncio
import websockets

async def send_video_feed(uri, stream_id, ip_cam_url):
    cap = cv2.VideoCapture(ip_cam_url)
    async with websockets.connect(uri) as websocket:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            encoded_frame = encode_frame(frame)
            await websocket.send(encoded_frame)
            await asyncio.sleep(0.03)  # Adjust the sleep time as needed

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_frame}"

if __name__ == "__main__":
    ip_cam_url = "http://192.168.69.36:4747/video"  # Replace with your IP camera URL
    uri = "ws://localhost:8000/ws/stream1"
    asyncio.run(send_video_feed(uri, "stream1", ip_cam_url))
