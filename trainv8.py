from ultralytics import YOLO
import pickle
import pprint
# Load a model

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

obj = pickle.load(open('dataset/labels.pkl'), 'wb')
with open("out.txt", 'a') as f:
    pprint.pprint(obj, stream=f)


# Train the model
# results = model.train(data='configs/deepdarts_d1.yaml', verbose=True)