from scipy.fftpack import fft2, dct, dst
import numpy as np
import cv2


def log_scale(mag_img):
    # switch to logarithmic scale
    mat_of_ones = np.ones(mag_img.shape, dtype=mag_img.dtype)
    cv2.add(mat_of_ones, mag_img, mag_img)
    cv2.log(mag_img, mag_img)


img = cv2.imread('img/noise.png', cv2.IMREAD_GRAYSCALE)
img_dct = dct(img)
img_fft = fft2(img)
img_dst = dst(img)
img2 = cv2.magnitude(img_fft.real, img_fft.imag)
log_scale(img2)
cv2.normalize(img2, img2, 0, 1, cv2.NORM_MINMAX)
# log_scale(img2)

cv2.imshow("fft real", img_fft.real)
cv2.imshow("fft imag", img_fft.imag)
cv2.imshow("dct", img_dct)
cv2.imshow("dst", img_dst)
cv2.imshow("fft", img2)
cv2.waitKey()
