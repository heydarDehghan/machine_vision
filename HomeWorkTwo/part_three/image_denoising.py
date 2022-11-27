import numpy as np
import cv2
from matplotlib import pyplot as plt
from basic_function import low_pass_filter_mask, add_gaussian_noise

# read image
img = cv2.imread('data/lena.png', 0)
img = add_gaussian_noise(img)
fft2 = np.fft.fft2(img)
fft2_shift = np.fft.fftshift(fft2)

low_pass_filter = fft2_shift * low_pass_filter_mask(50, img_shape=img.shape)
low_pass_image_result = np.fft.ifftshift(low_pass_filter)
inverse_low_pass_image_result = np.fft.ifft2(low_pass_image_result)

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(np.abs(inverse_low_pass_image_result), cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])

plt.show()
plt.savefig('result/noise.png')
