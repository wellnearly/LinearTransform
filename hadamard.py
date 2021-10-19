from scipy.linalg import hadamard
import numpy as np
import cv2

import numpy as np

def order_hadamard_matrix(had_mat):
    # вычисляем количество смены знаков в каждой строке
    sign_arr = []
    for str in had_mat:
        sign_arr.append(len(np.where(str[:-1] * str[1:] < 0)[0]))
    # print(sign_arr)

    # получаем упорядоченную матрицу Адамара
    swap_mat = had_mat.copy()
    had_mat[sign_arr] = swap_mat
    return had_mat



img = cv2.imread('img/girl2.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("original",img)

# преобразование применяется только для степеней двойки,
# поэтому находим ближайшие и дополняем изображение нулями
rows, cols = img.shape
m = 1<<(rows-1).bit_length()
n = 1<<(cols-1).bit_length()
complexI = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=0)

# planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
# complexI = cv2.merge(planes)
# print(complexI.shape)
print(m)
# создаем матрицу Адамара
had_mat_row = hadamard(m)
had_mat_col = hadamard(n)

# выполняем неупорядоченное преобразование Адамара
result = had_mat_row.dot(complexI).dot(had_mat_col) / (n*n)
cv2.imshow("natural Hadamard matrix transform", result.astype(np.uint8))

had_mat_row = order_hadamard_matrix(had_mat_row)
had_mat_col = order_hadamard_matrix(had_mat_col)
# print(had_mat_col)
# print(had_mat_row)
# выполняем упорядоченное преобразование Адамара
result = had_mat_row.dot(complexI).dot(had_mat_col) / (n*n)
cv2.imshow("ordered Hadamard matrix transform", result.astype(np.uint8))
# print(result)

#обнуляем нижнюю правую четверть матрицы, чтобы проиллюстрировать сжатие
lx, ly = result.shape
x, y = np.ogrid[0:lx,0:ly]
coeff = 2
mask = (x >= (lx >> coeff)) & (y >= (ly >> coeff))
result[mask] = 0

compressed_img = had_mat_row.dot(result).dot(had_mat_col)
cv2.imshow("compressed",compressed_img[:rows,:cols].astype(np.uint8))
cv2.waitKey()