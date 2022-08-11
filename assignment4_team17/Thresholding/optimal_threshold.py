import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread

def optimal_threshold(threshold,NumpyImage):
    histogram,bins = np.histogram(NumpyImage,range(257))

    # Cut distribution into 2 areas 
    background_histogram = histogram[:threshold]
    foreground_histogram = histogram[threshold:]
    
    # Compute the two centroids of the two different areas 
    mean1 = (background_histogram*np.arange(0,threshold)).sum()/background_histogram.sum()
    mean2 = (foreground_histogram*np.arange(threshold,len(histogram))).sum()/foreground_histogram.sum()
    
    # Compute the new threshold
    new_threshold = int(np.round((mean1+mean2)/2))
    
    if( new_threshold != threshold) : return optimal_threshold(new_threshold,NumpyImage)
    return new_threshold


def get_optimalThreshold_output(img):
    GrayImage = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    NumpyImage=np.array(GrayImage)
    start=time.time()
    Optimal_Threshold = optimal_threshold(threshold=100,NumpyImage=NumpyImage)
    end = time.time()
    total_time = end-start
    print("Computation_time of Optimal_Global_Thresholding function=",total_time)

    Image_After_Applying_Optimal_Threshold = GrayImage > Optimal_Threshold

    plt.imshow(Image_After_Applying_Optimal_Threshold, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('optimal_threshold.jpg',bbox_inches='tight',pad_inches = 0)
# get_optimalThreshold_output("Images/skull.jpg")