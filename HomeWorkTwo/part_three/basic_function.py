import numpy as np


def calculate_distance(x_one, x_two):
    return np.sqrt((x_one[0] - x_two[0]) ** 2 + (x_one[0] - x_two[0]) ** 2)


def high_pass_filter_mask(param, img_shape):
    rows, cols = img_shape[:2]
    mask = np.zeros(img_shape[:2])
    crow, ccol = int(rows / 2), int(cols / 2)  # center
    center = (crow, ccol)
    for x in range(rows):
        for y in range(cols):
            mask[x, y] = 1 - np.exp(((-calculate_distance((x, y), center) ** 2) / (2 * param ** 2)))
    return mask


def low_pass_filter_mask(param, img_shape):
    rows, cols = img_shape[:2]
    mask = np.zeros(img_shape[:2])
    crow, ccol = int(rows / 2), int(cols / 2)  # center
    center = (crow, ccol)
    for x in range(rows):
        for y in range(cols):
            mask[x, y] = np.exp(((-calculate_distance((x, y), center) ** 2) / (2 * param ** 2)))
    return mask


def add_gaussian_noise(X_img):
    row, col = X_img.shape
    mu, sigma = 0, 0.1  # Centre of the distribution, Standard deviation of the distribution
    # creating a noise with the same dimension as the dataset (2,2)
    noise = 500 * np.random.normal(mu, sigma, [row, col])
    noisy = X_img + noise

    return noisy
