
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps

def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.show()
plot.i = 0

def Read_GrayScale_Image(img:np.array):
    img = Image.open(img)
    img = ImageOps.grayscale(img)
    img = img.resize(size=(512, 512))

    return img


def Calculate_Size_After_Applying_Kernel(img_size: int, kernel_size: int) -> int:
    num_pixels = 0

    for i in range(img_size):
        added = i + kernel_size
        if added <= img_size:
            num_pixels += 1

    return num_pixels


def Convolution(img: np.array, kernel: np.array) -> np.array:
    Image_size = Calculate_Size_After_Applying_Kernel(
        img_size=img.shape[1],
        kernel_size=kernel.shape[1]
    )
    k = kernel.shape[0]

    convolved_img = np.zeros(shape=(Image_size, Image_size))

    for i in range(Image_size):
        for j in range(Image_size):
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img


def display_image(image_to_display, type):

    if type == "Gray":
        imgplot = plt.imshow(image_to_display, cmap='gray')
    else:
        imgplot = plt.imshow(image_to_display)

    plt.show()


def from_RGB_to_GS(image):

    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    imgGray = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)

    return imgGray


def save_image(image_name, image_to_save):

    mpimg.imsave(image_name, image_to_save)
##################################################################

