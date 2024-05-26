#!/bin/bash


test_names=("Shear_30_Scale_1_Rot_15" "Shear_30_Scale_1_Rot_8")
rotations=(15 0 0 25)
for i in "${!test_names[@]}"; do
    test_name="${test_names[i]}"
    rotation="${rotations[i]}"
    echo "Running test" 
    yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=False project=DeeperDarts name=$test_name batch=8 iou=0.3 cos_lr=True lr0=1E-2 optimizer=SGD augment=True save_conf=True plots=True fliplr=0 dropout=0 scale=1 shear=30 degrees=$rotation

done


