'''
*** Not Used ***
Used for testing the label extraction
May show the labels as flipped - yolo flips them back
'''
import matplotlib.pyplot as plt

def display_from_list(image_filepath, xys):
    img = plt.imread(image_filepath)
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])
    i = 0
    for xy in xys:
        if i < 4:
            plt.plot(1-xy[0], xy[1], 'bo') # calibration points
        else:
            plt.plot(xy[0], 1-xy[1], 'ro') # darts
        i+=1

    plt.show()

image_filepath ="dataset/images/d1_02_22_2020/IMG_4595.JPG"
xys= [[0.4405684754521964, 0.12782440284054228], [0.5594315245478035, 0.8721755971594577], [0.12758397932816537, 0.5593931568754034], [0.8724160206718347, 0.4406068431245966], [0.7945736434108527, 0.3867010974822466], [0.9308785529715762, 0.3382827630729503], [0.7493540051679587, 0.3750806972240155]]


def display_from_file(relative_img_path):
    image = f"datasets/test/images/{relative_img_path}.JPG"
    label = f"datasets/test/labels/{relative_img_path}.txt"
    
    img = plt.imread(image)
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])

    colours = ['cyan', 'y', 'r', 'lime', 'purple']

    with open(label, 'r') as f:
        lines = f.readlines()
        for line in lines:
            c,x,y,w,h = line.strip().split(' ')
            x = float(x)
            y = 1 - float(y)
            w = float(w)
            h = float(h)
            c = int(c)

            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2

            #plot box
            plt.plot([x1,x2], [y1,y1], colours[c], linewidth=0.5)
            plt.plot([x1,x1], [y1,y2], colours[c], linewidth=0.5)
            plt.plot([x1,x2], [y2,y2], colours[c], linewidth=0.5)
            plt.plot([x2,x2], [y1,y2], colours[c], linewidth=0.5)
            #plot cross hairs
            plt.plot([x,x], [y1,y2], colours[c], linewidth=0.5, linestyle=(0,(5,10))) # loosely dashed linestyle
            plt.plot([x1,x2], [y, y], colours[c], linewidth=0.5, linestyle=(0,(5,10)))

    plt.show()

image_path = "d1_02_04_2020/IMG_1081"
image_path = 'd1_02_04_2020/IMG_1087'
image_path = "d1_03_31_2020/IMG_6996"
display_from_file(image_path)