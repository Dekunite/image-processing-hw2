import cv2
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

def sobel_filter_start():
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
    #define vertical sobel filter matrix
    vertical_sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #define horizontal sobel filter matrix
    horizontal_sobel_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    #apply gaussian smoothing on the
    blurred_image = gaussianBlur(image)
    #apply vertical sobel mask
    sobelled_image_vertical = convolve(blurred_image, vertical_sobel_filter)
    #apply horizontal sobel mask
    sobelled_image_horizontal = convolve(blurred_image, horizontal_sobel_filter)

    return sobelled_image_vertical, sobelled_image_horizontal

def cv2_sobel_filters(image):
    sobel_vertical = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_horizontal = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    return sobel_vertical, sobel_horizontal

def gaussianBlur(image):
    #create a gaussian kernel
    kernel = get_gaussian_kernel(3)
    #apply gaussian kernel by convolution
    result = convolve(image, kernel)

    return result

def get_gaussian_kernel(size, sigma=1):
    #create a normal distrubution gaussian using the formula
    x = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gaussian = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gaussian, gaussian)
    result = kernel / np.sum(kernel)

    return result

def show_image(image, title):
    plt.imshow(image, cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()

#convolution method to apply masks on images
def convolve(image, kernel):
    k_rows, k_cols = kernel.shape
    rows, cols = image.shape
    result = np.zeros(image.shape)

    vertical = (k_rows - 1)
    horizontal= (k_cols - 1)
    padding_vert = int(vertical / 2)
    padding_hort = int(horizontal / 2)

    padded_shape_vert = rows + (2 * padding_vert)
    padded_shape_hort = cols + (2 * padding_hort)
    padded = np.zeros((padded_shape_vert, padded_shape_hort))

    padded[padding_vert:padded.shape[0] - padding_vert, padding_hort:padded.shape[1] - padding_hort] = image

    for row in range(rows):
        for col in range(cols):
            result[row, col] = np.sum(kernel * padded[row:row + k_rows, col:col + k_cols])

    return result

if __name__ == '__main__':
    sobel_filter_start()