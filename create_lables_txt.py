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

if __name__ == "__main__":
    label_path = 'dataset/labels'
    with open("all_labels.tsv", "r") as f:
        all = f.readlines()
        for line in all:
            directory, filename, bbox, xy = line.split("\t")
            if not os.path.isdir(f"{label_path}/{directory}"):
                os.mkdir(f"{label_path}/{directory}")
            xy = np.array(ast.literal_eval(xy))
            xywhc = get_bounding_boxes(xy,0.025)
            break
            # with open(f"label_path}/{directory}/{filename}", "w") as label_file:
            #     i = 0
            #     for xy in xys:
            #         i+=1
            #         label_file.write(f"{i} {xy[0]}")