import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_neigh(img,x,y,rows,cols):
    x+=1
    y+=1
    if x< rows and y < cols:
        pixel = img[x][y]
        if 0 < pixel < 100:
            res[x][y] = pixel
            check_neigh(img,x,y,rows,cols)


if __name__ == "__main__":
    img = cv2.imread("Lion.jpg",0)

    rows,cols = img.shape

    res = np.zeros(img.shape)
    for i in range(310,340):
        res[i][640] = 255

    plt.imshow(res, cmap="gray", aspect='auto')
    plt.show()

    x,y=320,640
    pixel = img[x][y]
    res[x][y] = pixel
    check_neigh(img,x,y,rows,cols)

    plt.imshow(res, cmap="gray", aspect='auto')
    plt.title("sadasads")
    plt.show()