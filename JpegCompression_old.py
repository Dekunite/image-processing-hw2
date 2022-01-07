import numpy as np
import cv2
import scipy.fftpack as fftpack

class jpeg:

    def __init__(self, im, quants):
        self.image = im
        self.quants = quants
        super().__init__()

    def encode_quant(self, quant):
        return (self.enc / quant).astype(np.int)

    def decode_quant(self, quant):
        return (self.encq * quant).astype(float)

    def encode_dct(self, bx, by):
        new_shape = (
            self.image.shape[0] // bx * bx,
            self.image.shape[1] // by * by,
            3
        )
        new = self.image[
              :new_shape[0],
              :new_shape[1]
              ].reshape((
            new_shape[0] // bx,
            bx,
            new_shape[1] // by,
            by,
            3
        ))
        return fftpack.dctn(new, axes=[1, 3], norm='ortho')

    def decode_dct(self, bx, by):
        return fftpack.idctn(self.decq, axes=[1, 3], norm='ortho'
                             ).reshape((
            self.decq.shape[0] * bx,
            self.decq.shape[2] * by,
            3
        ))

    def intiate(self, qscale, bx, by):
        quant = (
            (np.ones((bx, by)) * (qscale * qscale))
                .clip(-100, 100)  # to prevent clipping
                .reshape((1, bx, 1, by, 1))
        )
        self.enc = self.encode_dct(bx, by)
        self.encq = self.encode_quant(quant)
        self.decq = self.decode_quant(quant)
        self.dec = self.decode_dct(bx, by)
        img_bgr = ycbcr2rgb(self.dec)

        print(MSE_comparison(self.image, img_bgr))

        cv2.imwrite("{}/compressed_quant_{}_block_{}x{}.jpeg".format(
            output_dir, qscale, bx, by), img_bgr.astype(np.uint8))

def rgb2ycbcr(im_rgb):
    row, col, dim = im_rgb.shape
    im_rgb = im_rgb.astype(np.float32)
    red = im_rgb[:,:,0]
    testss = np.zeros(im_rgb.shape)
    green = im_rgb[:,:,1]
    blue = im_rgb[:,:,2]
    y=np.zeros((row,col))
    cb=np.zeros((row,col))
    cr=np.zeros((row,col))
    for i in range(row):
        for k in range(col):
            R = red[i][k]
            G = green[i][k]
            B = blue[i][k]

            y[i][k]= 0.299 * R + 0.587 * G + 0.114 * B#+128
            cb[i][k] = -0.1687* R -0.3313* G + 0.5* B #+ 128
            cr[i][k] = 0.5* R -0.4187* G -0.0813* B #+ 128
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    testss[:,:,0] = y
    testss[:,:,1] = cb
    testss[:,:,2] = cr
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0  # to [16/255, 240/255]
    return im_ycbcr
    #return testss.astype(np.float32)


def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)  # to [0, 1]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)  # to [0, 1]
    im_ycrcb = im_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb

def MSE_comparison(image, compressed):
    err = np.sum((image.astype("float") - compressed.astype("float")) ** 2)
    err /= float(image.shape[0] * image.shape[1])

    return err


if __name__ == "__main__":

    output_dir = "./"
    quant_size = 5
    block_size = 8

    im = cv2.imread("Valve_original_wiki.png")
    Ycr = rgb2ycbcr(im)
    obj = jpeg(Ycr, [5])
    quants = [quant_size]
    blocks = [(block_size, block_size)]
    for qscale in quants:
        for bx, by in blocks:
            obj.intiate(qscale, bx, by)

