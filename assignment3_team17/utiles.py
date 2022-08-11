from collections import ChainMap
import cv2
import matplotlib.pyplot as plt
import numpy as np


def gradient_x(imggray , k_size):

    if k_size == 5:
        kernel_x = np.array([
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-4, -2, 0, 2, 4],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2]])

    elif k_size == 3:
        kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]])

    return convolution( imggray, kernel_x )

def gradient_y(imggray, k_size):
    if k_size == 5:
        kernel_y = np.array([
            [-2, -2, -4, -2, -2],
            [-1, -1, -2, -1, -1],
            [ 0,  0,  0,  0,  0],
            [ 1,  1,  2,  1,  1],
            [ 2,  2,  4,  2,  2]])

    elif k_size == 3:
        kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]])

    return convolution( imggray, kernel_y )


def gaussian_filter(img, m, n, sigma):
    gaussian = np.zeros((m, n))
    m = m//2
    n = n//2
    for i in range(-m, m+1):
        for j in range(-n, n+1):
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(i**2+j**2)/(2*sigma**2))
            gaussian[i+m, j+n] = (1/x1)*x2
    
    return convolution( img, gaussian )


def convolution(image, kernel, average=False):
    if len(image.shape) == 3:
        # print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    # print("Kernel Shape : {}".format(kernel.shape))


    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image


    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    # print("Output Image size : {}".format(output.shape))

    return output

def plot_image( img ):
    # plt.figure( figsize = ( 6, 6 ) )
    # plt.imshow( img, cmap='gray'  )
    # plt.axis( 'off' )
    # plt.show( )
    plt.imshow(img,cmap='gray')
    plt.axis('off')
    plt.savefig('Harris_Image.jpg',bbox_inches='tight',pad_inches = 0)


def from_RGB_to_GS( image ):
    R, G, B = image[ :, :, 0], image[ :, :, 1], image[ :, :, 2]
    imgGray = ( 0.2989 * R + 0.5870 * G + 0.1140 * B ).astype( np.uint8 )
    return imgGray


