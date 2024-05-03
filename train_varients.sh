#!/bin/bash

test_names=("adam_dropout_0.1" "adam_droput_0.3" "adam_dropout_0.4" "SGD" "RMSProp" "RAdam")
droputs=(0.1 0.3 0.4 0 0 0)
optimizers=("Adam" "Adam" "Adam" "SGD" "RMSProp" "RAdam")

# Execute each command in the list
for i in "${!test_names[@]}"; do
    test_name = "${test_names[i]}"
    droput = "${droputs[i]}"
    optimzer = "${optimzers[i]}"

    echo "yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=True project=DeeperDarts name=$test_name batch=8 iou=0.3 cos_lr=True lr0=1E-3 optimizer=$optimizer augment=True save_conf=True plots=True dropout=$dropout"
    python predictv8.py  # Run the Python script
done
