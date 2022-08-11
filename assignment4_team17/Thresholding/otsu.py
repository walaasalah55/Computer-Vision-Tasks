import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
import cv2
import time

def otsu_threshold(NumpyImage):
    histogram, bins = np.histogram(NumpyImage.flatten(), range(257))

    # Calculate the Probability histogram
    Probability_Histogram = histogram/histogram.sum()

    """ create empty arrays to append the calculated
    variances on it.
    """
    Within_Variance = np.zeros(len(Probability_Histogram))
    Between_Variance = np.zeros(len(Probability_Histogram))
    Separability = np.zeros(len(Probability_Histogram))

    """ iterate on all possible thresholds and calculate
    the corresponding values of within&between variances to find
    the best threshold at which we have max value of between variance
    and the min value of within variance.
    """
    for threshold in range(1, len(Probability_Histogram)-1):
        # calculate the weights of background&foreground pixels
        weight_of_background = Probability_Histogram[:threshold].sum()
        weight_of_foreground = Probability_Histogram[threshold:].sum()

        # calculate the means of background&foreground pixels
        Mean_of_background = (np.arange(0, threshold)*Probability_Histogram[:threshold]).sum()/weight_of_background
        Mean_of_foreground = (np.arange(threshold, len(Probability_Histogram))*Probability_Histogram[threshold:]).sum()/weight_of_foreground

        s1 = (((np.arange(0, threshold)-Mean_of_background)**2) *
              Probability_Histogram[:threshold]).sum()/weight_of_background
        s2 = (((np.arange(threshold, len(Probability_Histogram))-Mean_of_foreground)
              ** 2)*Probability_Histogram[threshold:]).sum()/weight_of_foreground

        # Intra-class
        within_var = weight_of_background*s1+weight_of_foreground*s2

        # Inter-class
        between_var = weight_of_background*weight_of_foreground * \
            ((Mean_of_foreground-Mean_of_background)**2)

        # Separability
        Within_Variance[threshold] = within_var
        Between_Variance[threshold] = between_var

    Separability[1:-1] = Between_Variance[1:-1]/Within_Variance[1:-1]
    Best_thresold = np.argmax(Separability)

    return Best_thresold

def get_otsu_output(imgPath):
    GrayImage = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    NumpyImage = np.array(GrayImage)

    start=time.time()
    threshold= otsu_threshold(NumpyImage)
    end = time.time()
    total_time = end-start
    print("Computation_time of Otsu_Global_Thresholding function=",total_time)
    print("Otsu threshold : %d" % threshold)

    # plt.imshow(GrayImage, cmap=plt.cm.gray, vmin=0, vmax=255)
    # plt.subplot(1, 2, 2)
    plt.imshow(GrayImage > threshold, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('otsu.jpg',bbox_inches='tight',pad_inches = 0)

get_otsu_output('Images/skull.jpg')