#!/bin/bash

# test_names=("adam_dropout_0.1" "adam_dropout_0.3" "adam_dropout_0.4" "SGD" "RMSProp" "RAdam")

test_names=("Scale_1" "Shear_20" "Rot_15" "Scale_0.7" "Shear_30" "Rot_25")
scales=(1 0 0 0.7 0)
shears=(0 20 0 0 30 0)
rotations=(0 0 15 0 0 25)
for i in "${!test_names[@]}"; do
    test_name="${test_names[i]}"
    scale="${scales[i]}"
    shear="${shears[i]}"
    rotation="${rotations[i]}"
    echo "Running test" 
    yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=False project=DeeperDarts name=$test_name batch=8 iou=0.3 cos_lr=True lr0=1E-2 optimizer=SGD augment=True save_conf=True plots=True fliplr=0 dropout=0 scale=$scale shear=$shear degrees=$rotation

done
