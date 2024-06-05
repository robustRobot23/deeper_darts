'''
Training model was done in the terminal (in the conda environment)
Examples of terminal commands given here.
'''

# Train the model
#These examples were run in the terminal (in the conda environment)
# yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=True name=SecondRun batch=8

# yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=True name=DeeperDarts batch=8 iou=0.3 cos_lr=True lr0=1E-3 augment=True save_conf=True plots=True
# yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=True name=DeeperDarts batch=8 iou=0.3 cos_lr=True lr0=1E-3 optimizer=Adam augment=True save_conf=True plots=True 
# yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=True project=DeeperDarts name=droput_0.1 batch=8 iou=0.3 cos_lr=True lr0=1E-3 optimizer=Adam augment=True save_conf=True plots=True dropout=0.2