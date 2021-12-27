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

    # cv2 canny
    t1, t2 = 10, 20
    start = timer()
    cv2_canny_image = cv2_canny(image, t1, t2)
    end = timer()
    print("cv2 Canny Total Time: ", end - start)
    show_image(cv2_canny_image, ("cv2 Canny", t1, t2))
    t1, t2 = 100, 200
    cv2_canny_image = cv2_canny(image, t1, t2)
    show_image(cv2_canny_image, ("cv2 Canny", t1, t2))
    t1, t2 = 200, 500
    cv2_canny_image = cv2_canny(image, t1, t2)
    show_image(cv2_canny_image, ("cv2 Canny", t1, t2))

    image = cv2.imread("Valve_original_wiki.png", 0)
    # manual canny edge detect
    t1, t2 = 1, 2
    start = timer()
    canny_image = canny_edge_detect(image, t1, t2)
    end = timer()
    print("Manual Canny Total Time: ", end - start)
    show_image(canny_image, ("Manual Canny", t1, t2))
    t1, t2 = 10, 20
    cv2_canny_image = canny_edge_detect(image, t1, t2)
    show_image(cv2_canny_image, ("cv2 Canny", t1, t2))
    t1, t2 = 20, 50
    cv2_canny_image = canny_edge_detect(image, t1, t2)
    show_image(cv2_canny_image, ("cv2 Canny", t1, t2))


def cv2_canny(image, thresh1, thresh2):
    result = cv2.Canny(image, thresh1, thresh2)
    return result


def manual_sobel_edge_detect(image, kernel_size):
    sobelled_image_vertical, sobelled_image_horizontal = manual_sobel_filter(image, kernel_size)

    gradient_magnitude = np.sqrt(
        np.square(sobelled_image_vertical) + np.square(sobelled_image_horizontal))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    # calculating gradient direction
    gradient_direction = np.arctan2(sobelled_image_horizontal, sobelled_image_vertical)
    # convert radian to degree
    gradient_direction = np.rad2deg(gradient_direction)
    # rad2deg returns -180 to 180, add 180 to make thresholds 0 to 360
    gradient_direction += 180

    return gradient_magnitude, gradient_direction


def manual_sobel_filter(image, kernel_size):
    vertical_sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_sobel_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    blurred_image = gaussianBlur(image, kernel_size)
    sobelled_image_vertical = convolution(blurred_image, vertical_sobel_filter)
    sobelled_image_horizontal = convolution(blurred_image, horizontal_sobel_filter)

    return sobelled_image_vertical, sobelled_image_horizontal


def canny_edge_detect(image, t1, t2):
    magnitude_gradient, direction = manual_sobel_edge_detect(image, 3)

    result = non_maxima_suppression(magnitude_gradient, direction)
    result = threshold(result, t1, t2)

    result = hysteresis(result)

    return result


# to remove duplicate edges
def non_maxima_suppression(magnitude_gradient, direction):
    rows, columns = magnitude_gradient.shape

    # create 0 vector as start, all black image
    result = np.zeros(magnitude_gradient.shape)
    pi_degree = 180

    for row in range(1, rows - 1):
        for col in range(1, columns - 1):
            current_dir = direction[row, col]

            if (0 <= current_dir < pi_degree / 8) or (2 * pi_degree >= current_dir >= 15 * pi_degree / 8):
                neighbour1 = magnitude_gradient[row, col - 1]
                neighbour2 = magnitude_gradient[row, col + 1]

            elif (pi_degree / 8 <= current_dir < 3 * pi_degree / 8) or (11 * pi_degree / 8 >= current_dir > 9 * pi_degree / 8):
                neighbour1 = magnitude_gradient[row + 1, col - 1]
                neighbour2 = magnitude_gradient[row - 1, col + 1]

            elif (3 * pi_degree / 8 <= current_dir < 5 * pi_degree / 8) or (13 * pi_degree / 8 >= current_dir > 11 * pi_degree / 8):
                neighbour1 = magnitude_gradient[row - 1, col]
                neighbour2 = magnitude_gradient[row + 1, col]

            else:
                neighbour1 = magnitude_gradient[row - 1, col - 1]
                neighbour2 = magnitude_gradient[row + 1, col + 1]

            if magnitude_gradient[row, col] >= neighbour1 and magnitude_gradient[
                row, col] >= neighbour2:
                result[row, col] = magnitude_gradient[row, col]

    return result


def threshold(image, low, high):
    # white
    white = 255
    gray = 45
    # all black image
    output = np.zeros(image.shape)

    gray_rows, gray_cols = np.where((image <= high) & (image >= low))
    white_rows, white_cols = np.where(image >= high)

    output[gray_rows, gray_cols] = gray
    output[white_rows, white_cols] = white

    return output


def hysteresis(image):
    gray = 45
    image_row, image_col = image.shape

    top_to_bottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == gray:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[
                    row - 1, col + 1] == 255 or top_to_bottom[row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == gray:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[
                    row - 1, col + 1] == 255 or bottom_to_top[row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == gray:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[
                    row - 1, col + 1] == 255 or right_to_left[row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == gray:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[
                    row - 1, col + 1] == 255 or left_to_right[row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255

    return final_image


def show_image(image, title):
    plt.imshow(image, cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()


def gaussianBlur(image, kernel_size):
    kernel = get_gaussian_kernel(kernel_size)
    result = convolution(image, kernel)

    return result


def get_gaussian_kernel(size, sigma=1):
    x = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gaussian = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gaussian, gaussian)
    result = kernel / np.sum(kernel)

    return result


if __name__ == '__main__':
    start()
