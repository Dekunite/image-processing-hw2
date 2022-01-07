import cv2
import numpy as np
import scipy.fftpack as fftpack
from math import log10, sqrt

def jpeg_compression_start():
    image = cv2.imread("Valve_original_wiki.png")
    block_size_x = 8
    block_size_y = 8
    quantization_matrix =  np.array([[80,60,50,80,120,200,255,255],
                                     [55,60,70,95,130,255,255,255],
                                     [70,65,80,120,200,255,255,255],
                                     [70,85,110,145,255,255,255,255],
                                     [90,110,185,255,255,255,255,255],
                                     [120,175,255,255,255,255,255,255],
                                     [245,255,255,255,255,255,255,255],
                                     [255,255,255,255,255,255,255,255]])
    #reshape q matrix to same dimensions ac discrete cosign result
    quantization_matrix_reshaped = (quantization_matrix.reshape((1, block_size_x, 1, block_size_y, 1)))

    jpeg = jpeg_compression(image,block_size_x,block_size_y,quantization_matrix_reshaped)
    jpeg =jpeg.astype(np.uint8)
    cv2.imwrite("compressed_valve.png", jpeg)
    compare_images(image, jpeg)

def jpeg_compression(image, block_size_x, block_size_y,q_table):
    #reshape image for discrete cosign transform
    reshaped = image.reshape((image.shape[0] // block_size_x,block_size_x,image.shape[1] // block_size_y,
                              block_size_y,image.shape[2]))
    #perform discrete cosign transform on image using the axis 1 and 3 (block sizes)
    dctn_result = fftpack.dctn(reshaped, axes=[1,3],norm='ortho')

    #divide every value by quantization table and round it to nearest integer
    #eliminate high frequency data
    quantized = (dctn_result / q_table).astype(int)

    #reverse quantization by multiplying values by quantization table
    quantize_reversed = (quantized * q_table).astype(float)

    #perform inverse discrete cosign transform on image using the axis 1 and 3 (block sizes)
    decoded = fftpack.idctn(quantize_reversed, axes=[1, 3], norm='ortho')
    #reshape image back to its original shape
    result = decoded.reshape((quantize_reversed.shape[0] * block_size_x, quantize_reversed.shape[2] * block_size_y, 3))

    return result

def compare_images(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    print("Mean squared error: ", mse)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print("PSNR: ", psnr)

if __name__ == '__main__':
    jpeg_compression_start()