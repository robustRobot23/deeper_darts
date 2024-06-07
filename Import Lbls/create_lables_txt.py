'''
Create individual label text files from unpickled lablels.pkl
'''

import os
import numpy as np
import ast 

def get_bounding_boxes(xy, size):
    '''
    Code from DeepDarts
    '''

    xy[((xy[:, 0] - size / 2 <= 0) |
        (xy[:, 0] + size / 2 >= 1) |
        (xy[:, 1] - size / 2 <= 0) |
        (xy[:, 1] + size / 2 >= 1)), -1] = 0
    xywhc = []
    for i, _xy in enumerate(xy):
        if i < 4:
            cls = i + 1
        else:
            cls = 0
        if _xy[-1]:  # is visible
            xywhc.append([_xy[0], _xy[1], size, size, cls])
    xywhc = np.array(xywhc)
    return xywhc

def create_label_lines(xywhcs):
    '''
    Extract required label infromation for yolov8
    '''
    labels = ''
    for xywhc in xywhcs:
        x,y,w,h,c = xywhc
        if (x < 0) or (y < 0):
            print(f"grr")
        labels += f"{c} {x} {y} {w} {h}\n"
    labels = labels[:-1] #remove final newline char
    return labels

def save_labels_as_txt_file(labels, label_path, directory,filename):
    '''
    Write the labels to corresponding txt file
    '''
    file_dir = f"{label_path}/{directory}"
    filepath = f"{file_dir}/{filename}.txt"
    if not os.path.isdir(file_dir):
        # print(f"dir '{file_dir}' doesn't exist, making it")
        os.mkdir(file_dir)

    with open(filepath, 'w') as f:
         f.writelines(labels)

if __name__ == "__main__":
    label_path = 'datasets/labels'
    with open("all_labels.tsv", "r") as f: #all_labels.tsv is made by "unpickle.py"
        all = f.readlines()
        for line in all:
            directory, filename, bbox, xy = line.split("\t")
            filename = filename.split('.')[0]
            
            xy = np.array(ast.literal_eval(xy))
            xywhc = get_bounding_boxes(xy,0.025)
            labels = create_label_lines(xywhc)
            save_labels_as_txt_file(labels, label_path, directory,filename)
