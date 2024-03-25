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
    image = f"dataset/images/{relative_img_path}.JPG"
    label = f"dataset/labels/{relative_img_path}.txt"
    
    img = plt.imread(image)
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])

    colours = ['cyan', 'y', 'r', 'lime', 'purple']

    with open(label, 'r') as f:
        lines = f.readlines()
        for line in lines:
            c,x1,y1,x2,y2 = line.strip().split(' ')
            c = int(c)
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            mid_x = (x1+x2)/2
            mid_y = (y1+y2)/2
            #plot box
            plt.plot([x1,x2], [y1,y1], colours[c], linewidth=0.5)
            plt.plot([x1,x1], [y1,y2], colours[c], linewidth=0.5)
            plt.plot([x1,x2], [y2,y2], colours[c], linewidth=0.5)
            plt.plot([x2,x2], [y1,y2], colours[c], linewidth=0.5)
            #plot cross hairs
            plt.plot([mid_x,mid_x], [y1,y2], colours[c], linewidth=0.5, linestyle=(0,(5,10))) # loosely dashed linestyle
            plt.plot([x1,x2], [mid_y, mid_y], colours[c], linewidth=0.5, linestyle=(0,(5,10)))

    plt.show()

image_path = "d1_02_04_2020/IMG_1081"
image_path = 'd1_02_04_2020/IMG_1087'
display_from_file(image_path)