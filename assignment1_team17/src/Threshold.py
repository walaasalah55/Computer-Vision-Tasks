import numpy as np
from utils import Read_GrayScale_Image, Convolution
from src.EdgeDetection import plot_two_images
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def Global_Thershold(image: np.array):

    threshold_value = 135
    image = Read_GrayScale_Image(image)
    image_After_Thershold = np.zeros(shape=(image.size[0], image.size[1]))
    Rows = image.size[0]
    Columns = image.size[1]
    for row in range(Rows):
        for col in range(Columns):
            if image.getpixel((col, row)) > threshold_value:
                image_After_Thershold[row, col] = 255
            else:
                image_After_Thershold[row, col] = 0

    plot_two_images(img1=image, img2=image_After_Thershold)


def Local_Thershold(image: np.array, mode):
    Kernel = np.zeros(shape=(3, 3))
    if mode == "Mean":
        Mean_Kernel = np.array([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ])
        Kernel = Mean_Kernel
    elif mode == "Gaussian":
        Gaussian_Kernel = np.array([
            [1/16, 1/8, 1/16],
            [1/8, 1/4, 1/8],
            [1/16, 1/8, 1/16]
        ])
        Kernel = Gaussian_Kernel

    Origin_Gray_img = Read_GrayScale_Image(image)
    Mean_image = Convolution(img=np.array(Origin_Gray_img), kernel=Kernel)
    Origin_Gray_img = Origin_Gray_img.resize(size=Mean_image.shape)

    image_After_Thershold = np.zeros(
        shape=(Origin_Gray_img.size[0], Origin_Gray_img.size[1]))
    Rows = Origin_Gray_img.size[0]
    Columns = Origin_Gray_img.size[1]
    for row in range(Rows):
        for col in range(Columns):
            if Origin_Gray_img.getpixel((col, row)) > Mean_image[row, col]:
                image_After_Thershold[row, col] = 255
            else:
                image_After_Thershold[row, col] = 0

    plot_two_images(img1=Origin_Gray_img, img2=image_After_Thershold)



