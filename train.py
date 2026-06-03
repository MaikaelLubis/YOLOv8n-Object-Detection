from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data = "data.yaml",
    epochs = 50, # epochs sekitar 100
    imgsz = 540,
    batch = 16,
    workers = 4,
    device = 'cpu'
)