import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (you can use 'yolov8n.pt' for fastest results)
model = YOLO("yolov8n.pt")  # 'n' = nano version; for speed on M1

# Load video
video_path = "./demo_crowd.mp4"  # change to your video file path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Filter results for 'person' class (class ID = 0 in COCO)
    person_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)

    # Annotate frame
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"People Count: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display
    cv2.imshow("Crowd Counter", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
