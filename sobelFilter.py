import cv2
import matplotlib.pyplot as plt
from convolution import convolution
import numpy as np
from timeit import default_timer as timer

from ManualConvolution import get_padding_width_per_side
from ManualConvolution import add_padding_to_image
from ManualConvolution import convolve

def start():
    # read image
    image = cv2.imread("Valve_original_wiki.png", 0)
    if image is None:
        print("No image available")
        return "No image available"

    #cv2 sobel filters
    start = timer()
    cv2_vertical, cv2_horizontal = cv2_sobel_filters(image)
    end = timer()
    print("cv2 Sobel Total Time: ", end - start)
    show_image(cv2_vertical, "cv2 vertical sobel")
    show_image(cv2_horizontal, "cv2 horizontal sobel")

    image = cv2.imread("Valve_original_wiki.png", 0)
    #manual sobel filters
    start = timer()
    vertical, horizontal = manual_sobel_filter(image)
    end = timer()
    print("Manual Sobel Total Time: ", end - start)
    show_image(vertical, "Manual Sobel Vertical")
    show_image(horizontal, "Manual Sobel Horizontal")

def manual_sobel_filter(image):
    vertical_sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_sobel_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    blurred_image = gaussianBlur(image)
    sobelled_image_vertical = convolution(blurred_image, vertical_sobel_filter)
    sobelled_image_horizontal = convolution(blurred_image, horizontal_sobel_filter)

    return sobelled_image_vertical, sobelled_image_horizontal

def cv2_sobel_filters(image):
    sobel_vertical = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_horizontal = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    return sobel_vertical, sobel_horizontal

def gaussianBlur(image):
    kernel = get_gaussian_kernel(3)
    result = convolution(image, kernel)

    return result

def get_gaussian_kernel(size, sigma=1):
    x = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gaussian = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gaussian, gaussian)
    result = kernel / np.sum(kernel)

    return result

def show_image(image, title):
    plt.imshow(image, cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    start()