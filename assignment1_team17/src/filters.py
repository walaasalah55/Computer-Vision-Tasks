import numpy as np
from utils import *

# Read Image
image = mpimg.imread('Images/apple.jpeg')
gray_image = from_RGB_to_GS(image)

def gaussian_filter(m, n, sigma):
    gaussian = np.zeros((m, n))
    m = m//2
    n = n//2
    for i in range(-m, m+1):
        for j in range(-n, n+1):
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(i**2+j**2)/(2*sigma**2))
            gaussian[i+m, j+n] = (1/x1)*x2
    return gaussian
##################################################################

def median_filter(image):
    m = image.shape[0] 
    n = image.shape[1]

    # Traverse the image. For every 3X3 area,
    # find the median of the pixels and replace the center pixel by the median
    new_image = np.zeros([m, n])

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [image[i-1, j-1], image[i-1, j], image[i-1, j + 1], image[i, j-1], image[i, j],
                    image[i, j + 1], image[i + 1, j-1], image[i + 1, j], image[i + 1, j + 1]]

            temp = sorted(temp)
            new_image[i, j] = temp[4]

    new_image = new_image.astype(np.uint8)
    return new_image
######################################################################

def average_filter(mask_size, image):
    m = image.shape[0] 
    n = image.shape[1]

    # obtain mask
    mask = np.ones((mask_size, mask_size), np.float32)/9

    # Convolve the 3X3 mask over the image
    new_image = np.zeros([m, n])
    # print (new_image)

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = image[i-1, j-1]*mask[0, 0]
            +image[i-1, j]*mask[0, 1]+image[i-1, j + 1]*mask[0, 2]
            +image[i, j-1]*mask[1, 0] + image[i, j]*mask[1, 1]
            +image[i, j + 1]*mask[1, 2]+image[i + 1, j-1]*mask[2,0]
            +image[i + 1, j]*mask[2, 1]+image[i + 1, j + 1]*mask[2, 2]
            
            new_image[i, j] = temp

    new_image = new_image.astype(np.uint8)
    return new_image
########################################################################
# Highpass and Lowpass filters

def highpass_filter(image):
    img = Read_GrayScale_Image(image)
    highpass_kernel = np.array([[0, -1/4, 0],
                                [-1/4,  9, -1/4],
                                [0, -1/4, 0]])
    highpass_3x3 = Convolution(np.array(img), highpass_kernel)
    return highpass_3x3

'''
#high
([[-1, -1, -1],
[-1,  9, -1],
[-1, -1, -1]])

#low
([[1/9, 1/9, 1/9],
[1/9,  1/9, 1/9],
[1/9, 1/9, 1/9]])
'''

def lowpass_filter(image):
    img = Read_GrayScale_Image(image)
    lowpass_kernel =np.array([[0, 1/8, 0],
                             [1/8,  1/2, 1/8],
                             [0, 1/8, 0]])

    lowpass_3x3 = Convolution(np.array(img), lowpass_kernel)
    return lowpass_3x3
