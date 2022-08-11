from ast import Num
import cv2
import numpy as np
from matplotlib import pyplot as plt
from .optimal_threshold import optimal_threshold
from .otsu import otsu_threshold
import time
def LocalThresholding(Image: np.array, Horizontal_Blocks: int, Vertical_Blocks: int, ThresholdingFunction: str):
    """
     Input of the function:
     this function takes four arguments:
     Image, number of horizontal and vertical blocks that we want to divide the image to it
     and the name of thresholding techniques
     Output of the function:
     The image after applying the local threshold on it 

    """

    ImageHight, ImageWidth = Image.shape
    Horizontal_Step = int (ImageWidth / Horizontal_Blocks)
    Vertical_Step = int (ImageHight / Vertical_Blocks)
    Horizontal_Range = []
    Vertical_Range = []

    for i in range(0, Horizontal_Blocks):
        Horizontal_Range.append(Horizontal_Step * i)

    for i in range(0, Vertical_Blocks):
        Vertical_Range.append(Vertical_Step * i)

    Horizontal_Range.append(ImageWidth)
    Vertical_Range.append(ImageHight)

    Result = np.zeros((ImageHight, ImageWidth))

    for x in range(0, Horizontal_Blocks):
        for y in range(0, Vertical_Blocks):
            if ThresholdingFunction =='optimal_threshold':
                Result[Vertical_Range[y]:Vertical_Range[y + 1], Horizontal_Range[x]:Horizontal_Range[x + 1]] = optimal_threshold(threshold=25,NumpyImage=Image[Vertical_Range[y]:Vertical_Range[y + 1], Horizontal_Range[x]:Horizontal_Range[x + 1]])
            elif ThresholdingFunction =='otsu_threshold':
                Result[Vertical_Range[y]:Vertical_Range[y + 1], Horizontal_Range[x]:Horizontal_Range[x + 1]] = otsu_threshold(NumpyImage=Image[Vertical_Range[y]:Vertical_Range[y + 1], Horizontal_Range[x]:Horizontal_Range[x + 1]])

    return Result


def get_Local_Threshold_output(ImagePath,Horizontal_Blocks,Vertical_Blocks,Mode):
    start=time.time()
    GrayImage = cv2.imread(ImagePath,cv2.IMREAD_GRAYSCALE)
    NumpyImage=np.array(GrayImage)
    Image_After_Applying_Local_Threshold = LocalThresholding(NumpyImage,Horizontal_Blocks,Vertical_Blocks,Mode)
    # return NumpyImage>Image_After_Applying_Local_Threshold
    end = time.time()
    total_time = end-start
    print("Computation_time of Otsu_Local_Thresholding function=",total_time)

    plt.imshow(NumpyImage>Image_After_Applying_Local_Threshold, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('Local.jpg',bbox_inches='tight',pad_inches = 0)
