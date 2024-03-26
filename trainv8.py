from ultralytics import YOLO

# Load a model

model = YOLO('models/yolov8n.pt')  # load a pretrained model (recommended for training)

# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/csse/users/hev23/Documents/Computer Vision/Project/deeper_darts/datasets/val/data_custom.yaml', epochs=100, batch=8, imgsz=800, verbose=True)
#this line didn't work, so run from terminal instead:
# yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=True