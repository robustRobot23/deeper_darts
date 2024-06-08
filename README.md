Training a YOLOv8 model on dart board images anottated by [DeepDarts](https://github.com/wmcnally/deep-darts/tree/master)

## Setup
1. [Install Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a new conda environment with Python 3.7: ```$ conda create -n deeper-darts python==3.7```. Activate the environment: ```$ conda activate deeper-darts```
4. Clone this repo: ```$ git clone https://github.com/robustRobot23/deeper_darts.git```
5. Go into the directory and install the dependencies: ```$ cd deeper-darts && pip install -r requirements.txt```
6. Download the ```cropped_images.zip``` 800x800 cropped images directly from [IEEE Dataport](https://ieee-dataport.org/open-access/deepdarts-dataset). Extract and place in datasets/images/{train, val, or test}. (70:20:10) is a common split.
9. Download ```labels_pkl.zip```, and extract in the main directory.
10. Run ```Import Labels/unpickle.py```, then ```Import Labels/create_labels_txt.py```. This may require then moving the labels into the same directory structure as the corresponding image (just with labels/ not images/)



## Training
First change the train and val directories in data_custom.yaml to match your directory
To train enter into terminal ```yolo task=detect mode=train epochs=100 data=data_custom.yaml model=models/yolov8n.pt imgsz=800``` (can enter other params if desired), (this will download yolov8n.pt)
You may need to adjust the batch sizes to fit your total GPU memory. The default batch sizes are for 24 GB total GPU memory.

## Predicting
Open ```predictv8.py``` and enter the directory of the newly trained model, then run it.

