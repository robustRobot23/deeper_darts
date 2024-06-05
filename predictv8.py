import cv2
import os
import numpy as np
from datasets.annotate import draw, get_dart_scores
import random
import sys
import logging
import csv
import time

est_cal_pts_cnt = 0

def bboxes_to_xy(bboxes, max_darts=3):
    '''
    Modified from DeepDarts.
    Converts bounding box output from YOLOv8 of all classes to an xy centre point
    '''
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    num_darts = 0
    max_darts_exceeded = False

    for bbox in bboxes: 
        if int(bbox.cls) == 0 and not max_darts_exceeded: #bbox is around a dart, add dart centre to xy array
            dart_xywhn = bbox.xywhn[0] #centre coordinates
            # print(f"Dart found with xywhn: {dart_xywhn}")
            dart_x_centre = float(dart_xywhn[0])
            dart_y_centre = float(dart_xywhn[1])
            dart_xy_centre = np.array([dart_x_centre,dart_y_centre])
            # print(f"Dart centre xyn: {dart_xy_centre}")
            # print(f"Num_darts: {num_darts}")
            
            collumn = 4+num_darts
            try:
                xy[collumn,:2] = dart_xy_centre
            except IndexError:
                print(f"Couldn't add dart {num_darts+1}, index error")
            num_darts += 1
            if num_darts > max_darts:
                print("Max number of darts exceeded, ignoring any other detected darts")
                print("Need to add check for overlapping dart bounding boxes")
                max_darts_exceeded = True

        else:
            cal_xywhn = bbox.xywhn[0] #centre coordinates
            cal_x_centre = float(cal_xywhn[0])
            cal_y_centre = float(cal_xywhn[1])
            cal_xy_centre = np.array([cal_x_centre,cal_y_centre])

            collumn = int(bbox.cls)-1
            xy[collumn, :2] = cal_xy_centre #put calibration point in correct place in array


    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy


def est_cal_pts(xy):
    '''
    From DeepDarts
    Estimates any missed calibration points
    '''
    global est_cal_pts_cnt
    est_cal_pts_cnt += 1
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        # TODO: if len(missing_idx) > 1
        print('Missed more than 1 calibration point')
    return xy

def list_images_in_folder(folder_path):
    # List to store image file paths
    image_list = []

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a regular file and has an image extension
        if os.path.isfile(os.path.join(folder_path, filename)) and \
           filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # If the file is an image, add its path to the list
            image_list.append(os.path.join(folder_path, filename))

    return image_list

def get_label_xy(image_name, folder_path, max_darts=3):
    '''
    Gets the xy points from the label. Used for calculating 'actual' dart score.
    '''
    label_name = image_name.replace("JPG", "txt")
    label_path = f"{folder_path}/{label_name}".replace("images", "labels")
    # print(f"Label path: {label_path}")
    label_xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    num_darts = 0
    with open(label_path, 'r') as f:
        labels = f.readlines()
        # print(f"Length of labels: {len(labels)}")
        for label in labels:
            split_label = label.split(" ")
            class_num = int(split_label[0])
            x_centre = float(split_label[1])
            y_centre = float(split_label[2])
            xy_centre = np.array([x_centre, y_centre])
            
            if class_num == 0:
                label_xy[4+num_darts, :2] = xy_centre
                num_darts += 1
                # print(f"Dart {num_darts}: {xy_centre}")
            else:
                label_xy[class_num - 1, :2] = xy_centre

    # print(f"{num_darts} darts found in labels")
    label_xy[(label_xy[:, 0] > 0) & (label_xy[:, 1] > 0), -1] = 1
    return label_xy


def predict(model_directory):
    '''
    Used to predict dart location in 'image_folder_path' using the model in 'model_directory' 
    '''
    #path to test images
    image_folder_path = 'datasets/test/images/d1_03_31_2020'

    #make a list of all image paths
    images = list_images_in_folder(image_folder_path)

    # Directory to search for weights
    results_directory = 'DeeperDarts'
    
    best_weights_path = f'DeeperDarts/{model_directory}/weights/best.pt'

    print(f"Loading {best_weights_path}")
    #load model
    model = YOLO(best_weights_path)
    errors = []
    no_error_total = 0

    recent_results_img_dir = f"datasets/test/{model_directory}"
    os.makedirs(recent_results_img_dir, exist_ok=True)

    labeled_img_dir = f"{recent_results_img_dir}/scored_images"
    os.makedirs(labeled_img_dir, exist_ok=True)

    predicted_img_dir = f"{recent_results_img_dir}/predicted_images"
    os.makedirs(predicted_img_dir, exist_ok=True)

    log = logging.getLogger()
    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)

    # Set up logging
    log_dir = f"test_logs/{model_directory}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/processing_output.log"
    fileh = logging.FileHandler(log_file, 'a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    log.addHandler(fileh)
    print(f"Predicting {best_weights_path}")
    speeds = []
    
    for i in range(len(images)):
        image = images[i]
        image_name = images[i].split('/')[-1]
        # print(f"Processing {i}th image: '{image_name}'")

        results = model.predict(image)
        image_result = results[0]
        image_result.save(f"{predicted_img_dir}/{image_name}.JPG")
        boxes = image_result.boxes  
        speeds.append(image_result.speed['inference'])    
        xy = bboxes_to_xy(boxes) 
        xy = xy[xy[:, -1] == 1] #remove any empty rows
        predicted_score = get_dart_scores(xy,cfg, numeric=False)

        label_xy = get_label_xy(image_name, image_folder_path)
        label_xy = label_xy[label_xy[:, -1] == 1] #remove any empty rows
        actual_score = get_dart_scores(label_xy, cfg, numeric=False)

        error = sum(get_dart_scores(xy, cfg, numeric=True)) - sum(get_dart_scores(label_xy, cfg, numeric=True))
        errors.append(error)
        if error != 0:
            # print(f"Scoring board from image {image_name}")
            # print(f"Predicted Score: {predicted_score}")
            # print(f"Actual Score: {actual_score}") 
            # print(f"Error: {error}")
            logging.log(logging.WARN, f"Image {image_folder_path}/{image_name} had an error of {error}. Actual score: {actual_score}, Predicted score: {predicted_score}")
        else:
            no_error_total += 1

        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
        cv2.imwrite(osp.join(labeled_img_dir, image_name), img)


    avg_inference_time_ms = round(sum(speeds)/len(speeds), 2)

    abs_errors = map(abs, errors)

    avg_abs_error = round(sum(abs_errors)/len(errors),1)
    avg_error = round(sum(errors)/len(errors),1)
    PCS = round((100/len(errors))*no_error_total,1)

    test_name = model_directory

    #read training results.csv
    with open(f"{results_directory}/{model_directory}/results.csv") as f:
        epochs = 0
        reader = csv.reader(f)
        for row in reader:
            epochs = row[0]
    epochs = f"{epochs}".strip()

    # Append the results to a CSV file
    with open('test_results.csv', mode='a', newline='') as file:
        file.write(f"\n{test_name},{epochs},{PCS},{avg_error},{avg_abs_error},{avg_inference_time_ms}")

    print(f"Average absolute error:{sum(abs_errors)/len(errors)}")
    print(f"Average error: {sum(errors)/len(errors)}")
    print(f"PCS: {round((100/len(errors))*no_error_total,1)}%")
    print(f"Estimated Calibration Points {est_cal_pts_cnt} times")

if __name__ == '__main__':
    from ultralytics import YOLO
    print("imported yolo")

    from yacs.config import CfgNode as CN
    import os.path as osp

    #load configuration
    cfg = CN(new_allowed=True)
    cfg.merge_from_file('configs/deepdarts_d1.yaml')

    # model_directories = ["Rot_153","Rot_252","Scale_0.72","Scale_1","Shear_20","Shear_302"]
    # model_directories =   ["Shear_30_Scale_1_Rot_8", "Shear_30_Scale_1_Rot_152"]  
    results_dir = "DeeperDarts"
    model_directories = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    for model_directory in model_directories:
        predict(model_directory)