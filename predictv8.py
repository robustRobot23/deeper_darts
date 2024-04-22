import cv2
import os
import numpy as np
from dataset.annotate import draw, get_dart_scores

def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    dart_xys = np.array([])
    num_darts = 0
    max_darts_exceeded = False
    collumns = []
    # print(f"Length of bboxes: {len(bboxes)}")
    print(bboxes)
    print()
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
            xy[collumn,:2] = dart_xy_centre
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


if __name__ == '__main__':
    from ultralytics import YOLO
    from yacs.config import CfgNode as CN
    import os.path as osp

    cfg = CN(new_allowed=True)
    cfg.merge_from_file('configs/deepdarts_d1.yaml')
    
    # image_folder_path = 'some_test_imgs'
    # image_folder_path  = 'Personal Dart Board Images'
    # image_folder_path = '40_epoch_results'
    image_folder_path = 'datasets/val/images/d1_03_23_2020'
    images = list_images_in_folder(image_folder_path)
    print("imported yolo")
    model = YOLO('runs/detect/SecondRun/weights/best.pt')
    
    # for i in range(len(images)):
    for i in range(1):
        image = images[i]
        image_name = images[i].split('/')[-1]
        print(f"Processing {i}th image: '{image_name}'")

        result = model.predict(image)[0]
        boxes = result.boxes
        # print(boxes.data[1])
        # result.save(filename=image_name)
        # result.show() #display results to screen
        

        xy = bboxes_to_xy(boxes) 
        xy = xy[xy[:, -1] == 1] #remove any empty rows
        print(xy)
        score = get_dart_scores(xy,cfg, numeric=False)
        print(score)
        # error = sum(get_dart_scores(preds[i, :, :2], cfg, numeric=True)) - sum(get_dart_scores(xys[i, :, :2], cfg, numeric=True))

