import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def plot_image(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.show()


def Canny_edge_detector(image):
    Image =         cv2.imread(image)
    Gray_Image=     cv2.cvtColor(Image,cv2.COLOR_RGB2BGR)
    Blured_Image=   cv2.GaussianBlur(Gray_Image,(5, 5),-0.5)
    Canny=          cv2.Canny(Blured_Image,100,150)
  
    return Canny

