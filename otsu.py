import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise

def otsu_start():
    image = cv2.imread("otsu.jpeg", 0)
    images = ski_gauss(image)
    show_image(images, "Cv2 Gaussian noise")
    imagesss = gaussian_noise(image)
    show_image(imagesss, "Manuel Gaussian noise Var:500")
    show_image(otsus(images),"Otsu algorithm on cv2 gaussian")
    show_image(otsus(imagesss),"Otsu algorithm on manual gaussian Var:500")

def otsus(image):
    rows,cols = image.shape
    pixel_count = rows * cols
    mean = 1.0/pixel_count
    histogram, bins = np.histogram(image, np.arange(257))
    #0 to 255
    pixel_values = np.arange(256)
    result = image.copy()
    threshold = -999
    prev_variance = -999

    for bin in bins[1:-1]:
        mean_1 = np.sum(pixel_values[:bin]*histogram[:bin]) / float(np.sum(histogram[:bin]))
        mean_2 = np.sum(pixel_values[bin:]*histogram[bin:]) / float(np.sum(histogram[bin:]))
        prob1 = np.sum(histogram[:bin]) * mean
        prob2 = np.sum(histogram[bin:]) * mean

        variance = prob1 * prob2 * (mean_1 - mean_2) ** 2

        if variance > prev_variance:
            prev_variance = variance
            threshold = bin

    result[image > threshold] = 255
    result[image < threshold] = 0
    return result

def show_image(image, title):
    plt.imshow(image, cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()

def gaussian_noise(image):
    row, column = image.shape
    #create zero matrix
    result = np.zeros(image.shape, dtype=np.uint8)
    mean = 10
    #var = 100
    var = 500
    sigma = var**0.5
    #gaussin distribution
    gaussian = np.random.normal(mean, sigma, (row, column))
    gaussian = gaussian.reshape(row, column)
    result = image  +gaussian
    result = result.astype(np.uint8)
    return result

def ski_gauss(image):
    result = random_noise(image, mode='gaussian')
    #function returns between [0,1] range it is multiplied by 255
    result = np.array(255*result, dtype = 'uint8')
    return result

if __name__ == '__main__':
    otsu_start()