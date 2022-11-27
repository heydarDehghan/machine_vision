import numpy as np
import cv2
from matplotlib import pyplot as plt
from basic_function import high_pass_filter_mask

# read image
img = cv2.imread('data/lena.png', 0)
fft2 = np.fft.fft2(img)
fft2_shift = np.fft.fftshift(fft2)

high_pass_filter = fft2_shift * high_pass_filter_mask(30, img_shape=img.shape)
high_pass_image_result = np.fft.ifftshift(high_pass_filter)
inverse_high_pass_image_result = np.fft.ifft2(high_pass_image_result)

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(np.abs(inverse_high_pass_image_result), cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])

plt.show()
plt.savefig('result/edge_detection.png')
