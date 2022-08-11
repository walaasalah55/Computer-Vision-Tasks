import numpy as np
import matplotlib.pyplot as plt

from utils import Convolution, Read_GrayScale_Image
import matplotlib.image as mpimg
from utils import *

def plot_two_images(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')
    plt.show()

###########################################################
def Sobel_Edgd_Detection(img: np.array):
    sobel_Kernel_X = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_Kernel_Y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    #img = Read_GrayScale_Image(img)


    Vertical_Sobel = Convolution(img=np.array(img), kernel=sobel_Kernel_X)
    Horizontal_Sobel = Convolution(img=np.array(img), kernel=sobel_Kernel_Y)

    Sobel_Magntiude = np.sqrt(np.square(Vertical_Sobel) + np.square(Horizontal_Sobel))
    Sobel_Magntiude *= 255.0 / Sobel_Magntiude.max()
    
    Sobel_direction = np.arctan2(Horizontal_Sobel, Vertical_Sobel)
    Sobel_direction = np.rad2deg(Sobel_direction)
    Sobel_direction += 180

    #plot_two_images(img1=img, img2=Sobel_Magntiude)

    return Sobel_Magntiude, Sobel_direction

######################################################################
def Prewitt_Edge_Detection(img: np.array):
    Prewitt_Kernel_X = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    Prewitt_Kernel_Y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    img = Read_GrayScale_Image(img)

    Vertical_Prewitt = Convolution(img=np.array(img), kernel=Prewitt_Kernel_X)
    Horizontal_Prewitt = Convolution(
        img=np.array(img), kernel=Prewitt_Kernel_Y)

    PrewittEdge = np.sqrt(Horizontal_Prewitt**2 + Vertical_Prewitt**2)
    plot_two_images(img1=img, img2=PrewittEdge)

###################################################################
def Roberts_Edgd_Detection(img: np.array):
    Roberts_Kernel_X = np.array([
        [1, 0],
        [0, -1],
    ])

    Roberts_Kernel_Y = np.array([
        [0, 1],
        [-1, 0],
    ])

    img = Read_GrayScale_Image(img)

    Vertical_Roberts = Convolution(img=np.array(img), kernel=Roberts_Kernel_X)
    Horizontal_Roberts = Convolution(
        img=np.array(img), kernel=Roberts_Kernel_Y)

    RobertsEdge = np.sqrt(Horizontal_Roberts**2 + Vertical_Roberts**2)
    plot_two_images(img1=img, img2=RobertsEdge)
##########################################################################

#Sobel_Edgd_Detection('Images/Flower.jpeg')
# Prewitt_Edge_Detection('Flower.jpeg')
# Roberts_Edgd_Detection('Flower.jpeg')


