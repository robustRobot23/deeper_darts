#!/bin/bash

test_names=("adam_dropout_0.1" "adam_dropout_0.3" "adam_dropout_0.4" "SGD" "RMSProp" "RAdam")
dropouts=(0.1 0.3 0.4 0 0 0)
optimizers=("Adam" "Adam" "Adam" "SGD" "RMSProp" "RAdam")


# Execute each # Function to wait for a file to appear
wait_for_file() {
    local file=$1
    while [ ! -f "$file" ]; do
        sleep 1
    done
}

for i in "${!test_names[@]}"; do
    test_name="${test_names[i]}"
    dropout="${dropouts[i]}"
    optimizer="${optimizers[i]}"
    echo "yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800 verbose=True project=DeeperDarts name=$test_name batch=8 iou=0.3 cos_lr=True lr0=1E-3 optimizer=$optimizer augment=True save_conf=True plots=True dropout=$dropout"
    wait
done
