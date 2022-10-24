from PIL import Image as im
import numpy as np
import cv2

import random
import cv2


def add_salt_peper_noise(img):
    row, col = img.shape
    number_of_pixels = random.randint(3000, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255

    number_of_pixels = random.randint(3000, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0

    return img

def add_gaussian_noise(X_img):
    row, col = X_img.shape
    mu, sigma = 0, 0.1
    # creating a noise with the same dimension as the dataset (2,2)
    noise = 1000 * np.random.normal(mu, sigma, [row, col])
    noisy = X_img + noise

    return noisy


def convert_image_to_array(img):
    return np.array(img)


def read_image(path, gray=True):
    img = im.open(path)
    if gray:
        img = img.convert('L')
    return convert_image_to_array(img)


def convert_array_to_image(arr):
    pass
