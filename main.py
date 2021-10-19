import cv2
import numpy as np
from scipy.fftpack import fft2, dct, dst


def add_zeros_plate(img):
    # Add to the expanded another plane with zeros
    rows, cols = img.shape
    print("Высота:" + str(rows))
    print("Ширина:" + str(cols))
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)

    padded = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    return planes


def swap_quadrants(mag_img):
    mag_img_rows, mag_img_cols = mag_img.shape
    # crop the spectrum, if it has an odd number of rows or columns
    mag_img = mag_img[0:(mag_img_rows & -2), 0:(mag_img_cols & -2)]
    cx = int(mag_img_rows / 2)
    cy = int(mag_img_cols / 2)
    q0 = mag_img[0:cx, 0:cy]  # Top-Left - Create a ROI per quadrant
    q1 = mag_img[cx:cx + cx, 0:cy]  # Top-Right
    q2 = mag_img[0:cx, cy:cy + cy]  # Bottom-Left
    q3 = mag_img[cx:cx + cx, cy:cy + cy]  # Bottom-Right
    tmp = np.copy(q0)  # swap quadrants (Top-Left with Bottom-Right)
    mag_img[0:cx, 0:cy] = q3
    mag_img[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)  # swap quadrant (Top-Right with Bottom-Left)
    mag_img[cx:cx + cx, 0:cy] = q2
    mag_img[0:cx, cy:cy + cy] = tmp


def log_scale(mag_img):
    # switch to logarithmic scale
    mat_of_ones = np.ones(mag_img.shape, dtype=mag_img.dtype)
    cv2.add(mat_of_ones, mag_img, mag_img)
    cv2.log(mag_img, mag_img)


img = cv2.imread('img/girl.png', cv2.IMREAD_GRAYSCALE)
imgArr = [img]
for img in imgArr:
    # cv2.imshow("Input Image", img)
    #
    # planes = add_zeros_plate(img)
    # complexI = cv2.merge(planes)
    #
    # cv2.dft(complexI, complexI)  # this way the result may fit in the source matrix
    #
    # cv2.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    # cv2.magnitude(planes[0], planes[1], planes[0])  # planes[0] = magnitude
    # magI = planes[0]
    #
    # swap_quadrants(magI)
    # log_scale(magI)
    #
    # cv2.normalize(magI, magI, 0, 1, cv2.NORM_MINMAX)  # Transform the matrix with float values into a
    #
    # # Show the result
    # cv2.imshow("dft", magI)
    # cv2.waitKey()
    img_dct = dct(img)
    img_fft = fft2(img)
    img_dst = dst(img)
    img2 = cv2.magnitude(img_fft.real, img_fft.imag)
    swap_quadrants(img2)
    log_scale(img2)
    # cv2.normalize(img_dct, img_dct, 0, 1, cv2.NORM_MINMAX)
    # log_scale(img_dct)
    # log_scale(img_dst)
    cv2.normalize(img2, img2, 0, 1, cv2.NORM_MINMAX)
    print(img_dct)
    # cv2.normalize(img_dst, img_dst, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow("fft real", img_fft.real)
    cv2.imshow("fft imag", img_fft.imag)
    cv2.imshow("dct", img_dct)
    cv2.imshow("dst", img_dst)
    cv2.imshow("fft", img2)
    cv2.waitKey()




