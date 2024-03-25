#create individual label text files from unpickled lablels.pkl
import os
import numpy as np
import ast 

def get_bounding_boxes(xy, size):
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
    labels = ''
    for xywhc in xywhcs:
        x,y,w,h,c = xywhc

        # if c == 0: # darts
        #     x = 1 - x
        # else: #calibration points
        y = 1 - y

        y1 = round(y + h, 3)
        y2 = round(y - h, 3)
        x1 = round(x + w, 3)
        x2 = round(x - w, 3)
        labels += f"{int(c)} {x1} {y1} {x2} {y2}\n"
    labels = labels[:-1] #remove final newline char
    return labels

def save_labels_as_txt_file(labels, label_path, directory,filename):
    file_dir = f"{label_path}/{directory}"
    filepath = f"{file_dir}/{filename}.txt"
    if not os.path.isdir(file_dir):
        print(f"dir '{file_dir}' doesn't exist, making it")
        os.mkdir(file_dir)

    with open(filepath, 'w') as f:
         f.writelines(labels)

if __name__ == "__main__":
    label_path = 'dataset/labels'
    with open("all_labels.tsv", "r") as f:
        all = f.readlines()
        i = 0
        for line in all:
            directory, filename, bbox, xy = line.split("\t")
            filename = filename.split('.')[0]
            
            xy = np.array(ast.literal_eval(xy))
            xywhc = get_bounding_boxes(xy,0.025)
            labels = create_label_lines(xywhc)
            save_labels_as_txt_file(labels, label_path, directory,filename)
            i += 1
            if i == 10:
                break
