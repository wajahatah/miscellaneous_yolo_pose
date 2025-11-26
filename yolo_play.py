from ultralytics import YOLO

# Load YOLO model (use yolov8n.pt for speed or yolov8s.pt for better accuracy)
model = YOLO('yolo11n.pt')

# Run detection on webcam (0 = default camera)
model.predict(source=0, show=True, classes=[0, 56])