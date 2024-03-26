from ultralytics import YOLO

# Load a model

model = YOLO('models/yolov8n.pt')  # load a pretrained model (recommended for training)

# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/csse/users/hev23/Documents/Computer Vision/Project/deeper_darts/datasets/val/data_custom.yaml', epochs=100, batch=8, imgsz=800, verbose=True)