import cv2
import numpy as np
from scipy.fftpack import fft2, dct, dst, idct

# связь DCT и DFT
img = cv2.imread('img/girl.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("original", img)

rows, cols = img.shape
m = 2*rows
n = 2*cols
complexI = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=0)
complexI[0:rows, cols:n] = np.fliplr(complexI[0:rows, 0:cols])
complexI[rows:m, 0:cols] = np.flipud(complexI[0:rows, 0:cols])
complexI[rows:m, cols:n] = np.fliplr(complexI[rows:m, 0:cols])
# cv2.imshow("construct",complexI)
complex_fft_img = fft2(np.float32(complexI))
img_real = complex_fft_img.real
img_imag = complex_fft_img.imag
dct_img = cv2.dct(np.float32(img))

i, j = np.ogrid[0:256, 0:256]
m = np.zeros(shape=(rows, cols))
m[i, j] = 1/2 * (
        np.cos(np.pi*(i+j)/n)*img_real[i, j]
        - np.sin(np.pi*(i+j)/n)*img_imag[i, j]
        + np.cos(np.pi*(i-j)/n)*img_real[i, n-j-1]
        - np.sin(np.pi*(i-j)/n)*img_imag[i, n-j-1]
)
m[rows-i-1, j] = 1/2 * (
        np.sin(np.pi*(i+j)/n)*img_real[i, j]
        + np.cos(np.pi*(i+j)/n)*img_imag[i, j]
        + np.sin(np.pi*(i-j)/n)*img_real[i, n-j-1]
        + np.cos(np.pi*(i-j)/n)*img_imag[i, n-j-1]
)
m[i, rows-j-1] = 1/2 * (
        np.sin(np.pi*(i+j)/n)*img_real[i, j]
        + np.cos(np.pi*(i+j)/n)*img_imag[i, j]
        - np.sin(np.pi*(i-j)/n)*img_real[i, n-j-1]
        - np.cos(np.pi*(i-j)/n)*img_imag[i, n-j-1]
)
m[rows-i-1, rows-j-1] = 1/2 * (
        - np.cos(np.pi*(i+j)/n)*img_real[i, j]
        + np.sin(np.pi*(i+j)/n)*img_imag[i, j]
        + np.cos(np.pi*(i-j)/n)*img_real[i, n-j-1]
        - np.sin(np.pi*(i-j)/n)*img_imag[i, n-j-1]
)
m = m/n
# rotate = np.array([
#     [1,0,1,0],
#     [0,1,0,1],
#     [0,1,0,-1],
#     [-1,0,1,0]
# ])
# print(rotate)
# ortog = [
#     [np.cos(np.pi*(i+j)/n), -np.sin(np.pi*(i+j)/n), 0, 0],
#     [np.sin(np.pi*(i+j)/n), np.cos(np.pi*(i+j)/n), 0, 0],
#     [0, 0, np.cos(np.pi*(i-j)/n), -np.sin(np.pi*(i-j)/n)],
#     [0, 0, np.sin(np.pi*(i-j)/n), np.cos(np.pi*(i-j)/n)]
# ]
inverse_dct = cv2.dct(m, flags=cv2.DCT_INVERSE)
cv2.imshow("dct from dft inverse",inverse_dct.astype(np.uint8))

# print(inverse_dct)
cv2.waitKey()