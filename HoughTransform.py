import math
import cv2
import matplotlib.pyplot as plt
from convolution import convolution
import numpy as np
from timeit import default_timer as timer

def start():
    # read image
    image = cv2.imread("Valve_original_wiki.png", 0)
    if image is None:
        print("No image available")
        return "No image available"

    #cv2 sobel filters
    start = timer()
    cv2_hough_transform(image)
    end = timer()
    print("cv2 Sobel Total Time: ", end - start)

    image = cv2.imread("Valve_original_wiki.png", 0)
    #manual sobel filters
    start = timer()
    vertical, horizontal = manual_hough_transform(image)
    end = timer()
    print("Manual Sobel Total Time: ", end - start)
    show_image(vertical, "Manual Sobel Vertical")
    show_image(horizontal, "Manual Sobel Horizontal")

def cv2_hough_transform(image):
    cv2_hough_lines(image)
    cv2_hough_circles(image)

def cv2_hough_lines(image):
    # Apply edge detection method on the image
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    show_image(edges, "edege")

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for r, theta in lines[0]:
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # All the changes made in the input image are finally
    # written on a new image houghlines.jpg
    cv2.imshow('linesDetected.jpg', image)

def cv2_hough_circles(image):
    image = cv2.medianBlur(image, 5)
    cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    houghImageCircle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, param1=70, param2=60,
                                        minRadius=10, maxRadius=100)

    circles = np.uint16(np.around(houghImageCircle))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # show_image(cimg, "lines")
    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def manual_hough_transform(image):
    manual_hough_lines(image)
    manual_hough_circles(image)

def manual_hough_lines(image):
    print("ciliensrc")
def manual_hough_circles(image):
    print("circ")

def show_image(image, title):
    plt.imshow(image, cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    start()
