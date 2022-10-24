import cv2
import numpy as np
import pandas as pd
from basic_functions import read_image, convert_image_to_array, add_gaussian_noise, add_salt_peper_noise
from PIL import Image as im

if __name__ == '__main__':
    list_img = ['1.jpg', '2.jpg', '3.jpg']
    size_list = [3, 5, 7]
    q_part = 'part_three_b'

    for size in size_list:

        for item in list_img:
            data = read_image(f'data/{item}')

            data = add_salt_peper_noise(data)
            # data = add_gaussian_noise(data)

            noisi_image = im.fromarray(data)
            if noisi_image.mode != 'RGB':
                noisi_image = noisi_image.convert('RGB')
            noisi_image.save(f'result/part_three/{q_part}/img_with_noise_{item}')

            temp = np.zeros(data.shape)

            for i in range(data.shape[0]):
                if i + size <= data.shape[0]:
                    for j in range(data.shape[1]):
                        if j + size <= data.shape[1]:
                            w = data[i:i + size, j: j + size]
                            # temp[i, j] = np.mean(w)
                            temp[i, j] = np.median(w)

            temp = im.fromarray(temp)

            if temp.mode != 'RGB':
                temp = temp.convert('RGB')

            temp.save(f'result/part_three/{q_part}/img_size{size}_{item}')


