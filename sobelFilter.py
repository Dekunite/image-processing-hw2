import math

import cv2
import matplotlib.pyplot as plt
from convolution import convolution
import numpy as np


def sobelFilter(name):
    #read image
    image = cv2.imread("Valve_original_wiki.png",0)
    if image is None:
        print("No image available")
        return "No image available"

    sobelX = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    sobelY = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
    sobelXY = cv2.Sobel(image,cv2.CV_64F,1,1,ksize=3)

    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel = gaussian_kernel(3)
    newKernel = gkern(3,1)

    convImage = convolution(image, kernel)
    sobelledImage = convolution(convImage, filter)
    sobelledImageT = convolution(convImage, np.flip(filter.T, axis=0))

    show_image(image,"o")
    show_image(convolution(image, kernel),"o")
    show_image(convolution(image, kernel, True),"o")
    show_image(sobelledImage,"sobelled hort")
    show_image(sobelX,"x")
    show_image(sobelY,"y")
    show_image(sobelledImageT,"sobelled vert")

    gradient_magnitude = np.sqrt(np.square(sobelledImage) + np.square(sobelledImageT))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    show_image(gradient_magnitude, "gradiented")

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def show_image(image, title):
    plt.imshow(image,cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()

def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sobelFilter('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
