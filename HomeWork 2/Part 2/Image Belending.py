import numpy as np
import cv2
import matplotlib.pyplot as plt


def gaussian_pyramid(image):
    layer = image.copy()
    gp = [layer]
    for i in range(7):
        layer = cv2.pyrDown(layer)
        gp.append(layer)
    return gp


def laplacian_pyramid(gp):
    layer = gp[6]
    lp = [layer]
    for i in range(6, 0, -1):
        size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
        gauss_extend = cv2.pyrUp(gp[i], dstsize=size)
        laplacian = cv2.subtract(gp[i - 1], gauss_extend)
        lp.append(laplacian)
    return lp


def blending(lap_img1, lap_img2):
    apple_orange_pyramid = []
    n = 0
    for apple, orange in zip(lap_img1, lap_img2):
        n += 1
        row, col, channel = apple.shape
        laplacian = np.hstack((apple[:, 0:int(col / 2)], orange[:, int(col / 2):]))
        apple_orange_pyramid.append(laplacian)

    apple_orange_reconstruct = apple_orange_pyramid[0]
    for i in range(1, 7):
        size = (apple_orange_pyramid[i].shape[1], apple_orange_pyramid[i].shape[0])
        apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct, dstsize=size)
        apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)

    return apple_orange_reconstruct


def plot_image(img1, img2, blending):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Blending reconstruction')

    ax1.imshow(cv2.cvtColor(img1,cv2.COLOR_RGB2BGR))
    ax2.imshow(cv2.cvtColor(img2,cv2.COLOR_RGB2BGR))
    ax3.imshow(cv2.cvtColor(blending,cv2.COLOR_RGB2BGR))
    plt.show()


if __name__ == '__main__':
    apple = cv2.imread('Apple.png')
    orange = cv2.imread('Orange.png')

    gaussian_apple = gaussian_pyramid(apple)
    gaussian_orange = gaussian_pyramid(orange)

    laplacian_apple = laplacian_pyramid(gaussian_apple)
    laplacian_orange = laplacian_pyramid(gaussian_orange)

    print(len(laplacian_apple))

    reconstruct = blending(laplacian_apple, laplacian_orange)

    plot_image(apple, orange, reconstruct)
