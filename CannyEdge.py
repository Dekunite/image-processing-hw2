import cv2
import matplotlib.pyplot as plt
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
    show_image(cv2_canny_image, ("cv2 Canny", 'threshold bottom:', t1, 'threshold top:', t2))
    t1, t2 = 20, 50
    cv2_canny_image = cv2_canny(image, t1, t2)
    show_image(cv2_canny_image, ("cv2 Canny", 'threshold bottom:', t1, 'threshold top:', t2))
    t1, t2 = 50, 200
    cv2_canny_image = cv2_canny(image, t1, t2)
    show_image(cv2_canny_image, ("cv2 Canny", 'threshold bottom:', t1, 'threshold top:', t2))

    image = cv2.imread("Valve_original_wiki.png", 0)
    # manual canny edge detect
    t1, t2 = 1, 2
    start = timer()
    canny_image = canny_edge_detect(image, t1, t2, 3)
    end = timer()
    print("Manual Canny Total Time: ", end - start)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 3))
    canny_image = canny_edge_detect(image, t1, t2, 5)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 5))
    canny_image = canny_edge_detect(image, t1, t2, 7)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 7))
    t1, t2 = 10, 20
    canny_image = canny_edge_detect(image, t1, t2, 3)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 3))
    canny_image = canny_edge_detect(image, t1, t2, 7)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 5))
    canny_image = canny_edge_detect(image, t1, t2, 5)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 7))
    t1, t2 = 20, 50
    canny_image = canny_edge_detect(image, t1, t2, 3)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 3))
    canny_image = canny_edge_detect(image, t1, t2, 7)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 5))
    canny_image = canny_edge_detect(image, t1, t2, 5)
    show_image(canny_image, ("Manual Canny", 'threshold bottom:', t1, 'threshold top:', t2, 'sigma:', 7))


def cv2_canny(image, thresh1, thresh2):
    result = cv2.Canny(image, thresh1, thresh2)
    return result


def manual_sobel_edge_detect(image, kernel_size, sigma):
    vertical_sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_sobel_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    blurred_image = gaussianBlur(image, kernel_size, sigma)
    sobelled_image_vertical = convolve(blurred_image, vertical_sobel_filter)
    sobelled_image_horizontal = convolve(blurred_image, horizontal_sobel_filter)

    magnitude_gradient = np.sqrt(
        np.square(sobelled_image_vertical) + np.square(sobelled_image_horizontal))
    magnitude_gradient *= 255.0 / magnitude_gradient.max()

    # get direction of gradient
    direction = np.arctan2(sobelled_image_horizontal, sobelled_image_vertical)
    # converto to degrees
    direction = np.rad2deg(direction)
    direction += 180

    return magnitude_gradient, direction


def canny_edge_detect(image, t1, t2, sigma):
    magnitude_gradient, direction = manual_sobel_edge_detect(image, 3, sigma)

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

    top_to_bot = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bot[row, col] == gray:
                if top_to_bot[row, col + 1] == 255 or top_to_bot[row, col - 1] == 255 or top_to_bot[row - 1, col] == 255 or top_to_bot[
                    row + 1, col] == 255 or top_to_bot[row - 1, col - 1] == 255 or top_to_bot[row + 1, col - 1] == 255 or top_to_bot[
                    row - 1, col + 1] == 255 or top_to_bot[row + 1, col + 1] == 255:
                    top_to_bot[row, col] = 255
                else:
                    top_to_bot[row, col] = 0

    bot_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bot_to_top[row, col] == gray:
                if bot_to_top[row, col + 1] == 255 or bot_to_top[row, col - 1] == 255 or bot_to_top[row - 1, col] == 255 or bot_to_top[
                    row + 1, col] == 255 or bot_to_top[row - 1, col - 1] == 255 or bot_to_top[row + 1, col - 1] == 255 or bot_to_top[
                    row - 1, col + 1] == 255 or bot_to_top[row + 1, col + 1] == 255:
                    bot_to_top[row, col] = 255
                else:
                    bot_to_top[row, col] = 0

    r_to_l = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if r_to_l[row, col] == gray:
                if r_to_l[row, col + 1] == 255 or r_to_l[row, col - 1] == 255 or r_to_l[row - 1, col] == 255 or r_to_l[
                    row + 1, col] == 255 or r_to_l[row - 1, col - 1] == 255 or r_to_l[row + 1, col - 1] == 255 or r_to_l[
                    row - 1, col + 1] == 255 or r_to_l[row + 1, col + 1] == 255:
                    r_to_l[row, col] = 255
                else:
                    r_to_l[row, col] = 0

    l_to_r = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if l_to_r[row, col] == gray:
                if l_to_r[row, col + 1] == 255 or l_to_r[row, col - 1] == 255 or l_to_r[row - 1, col] == 255 or l_to_r[
                    row + 1, col] == 255 or l_to_r[row - 1, col - 1] == 255 or l_to_r[row + 1, col - 1] == 255 or l_to_r[
                    row - 1, col + 1] == 255 or l_to_r[row + 1, col + 1] == 255:
                    l_to_r[row, col] = 255
                else:
                    l_to_r[row, col] = 0

    result = top_to_bot + bot_to_top + r_to_l + l_to_r
    # limit at 255
    result[result > 255] = 255

    return result


def show_image(image, title):
    plt.imshow(image, cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()


def gaussianBlur(image, kernel_size, sigma):
    kernel = get_gaussian_kernel(kernel_size, sigma)
    result = convolve(image, kernel)

    return result


def convolve(image, kernel):
    k_rows, k_cols = kernel.shape
    rows, cols = image.shape
    result = np.zeros(image.shape)

    vert = (k_rows - 1)
    hort = (k_cols - 1)
    padding_vert = int(vert / 2)
    padding_hort = int(hort / 2)

    padded_shape_vert = rows + (2 * padding_vert)
    padded_shape_hort = cols + (2 * padding_hort)
    padded = np.zeros((padded_shape_vert, padded_shape_hort))

    padded[padding_vert:padded.shape[0] - padding_vert, padding_hort:padded.shape[1] - padding_hort] = image

    for row in range(rows):
        for col in range(cols):
            result[row, col] = np.sum(kernel * padded[row:row + k_rows, col:col + k_cols])

    return result


def get_gaussian_kernel(size, sigma=1):
    x = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gaussian = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gaussian, gaussian)
    result = kernel / np.sum(kernel)

    return result


if __name__ == '__main__':
    start()
