import cv2
import matplotlib.pyplot as plt
import numpy as np

import time

def plot_NCC_image(img1: np.array):
    plt.imshow(img1)
    plt.axis('off')
    plt.savefig('NCC_Matched_Image.jpg',bbox_inches='tight',pad_inches = 0)
def plot_SDD_image(img1: np.array):
    plt.imshow(img1)
    plt.axis('off')
    plt.savefig('SDD_Matched_Image.jpg',bbox_inches='tight',pad_inches = 0)

def Sum_Square_Difference(descriptors_1: np.ndarray, descriptors_2: np.ndarray):
    """
    This function takes two inputs, this two inputs are the descriptors array 
    which returned from the sift algorithm

    The output of this function is array of the matched features,
    This matched features are detected using method of sum square difference(SSD)

    """

    num_key_points1 = descriptors_1.shape[0]
    num_key_points2 = descriptors_2.shape[0]

    matches = []
    for KeyPoint1 in range(num_key_points1):
        distance = -np.inf
        matched_key_point = -1

        for KeyPoint2 in range(num_key_points2):

            ssd_value = 0
            for feature in range(descriptors_1.shape[1]):
                """
                Calculate the ssd between the features of the original image
                and the template image, if the ssd value is small(i.e around zero) 
                it means that this two features are the same 

                So we compare the ssd values of all features to get the smallest values
                """
                ssd_value += (descriptors_1[KeyPoint1, feature] - descriptors_2[KeyPoint2, feature]) **2

            if -ssd_value > distance:
                distance = -ssd_value
                matched_key_point = KeyPoint2

        cur = cv2.DMatch()
        cur.queryIdx = KeyPoint1
        cur.trainIdx = matched_key_point
        cur.distance = distance
        matches.append(cur)
    return matches


def Normalized_Cross_Correlation(descriptors_1: np.ndarray, descriptors_2: np.ndarray):
    """
    This function takes two inputs, this two inputs are the descriptors array 
    which returned from the sift algorithm

    The output of this function is array of the matched features,
    This matched features are detected using method of Normalized Cross Correlation(NCC)

    """

    num_key_points1 = descriptors_1.shape[0]
    num_key_points2 = descriptors_2.shape[0]

    matches = []
    for KeyPoint1 in range(num_key_points1):
        distance = -np.inf
        matched_key_point = -1

        for KeyPoint2 in range(num_key_points2):
            """
            we calculate the normalized cross correlation by calculating the cross
            correlation value, the square root of sum square of all features of the 
            original image and the square root of sum square of all features of the 
            template image 

            then to compute the ncc value we divide cross correlation over the two 
            square roots multiplied by each other 
            """
            cross_correlation = np.sum(np.multiply(descriptors_1[KeyPoint1], descriptors_2[KeyPoint2]))
            summation_of_main_descriptors = np.sqrt(np.sum(np.square(descriptors_1[KeyPoint1])))
            summation_of_tempelte_descriptors = np.sqrt(np.sum(np.square(descriptors_2[KeyPoint2])))

            ncc = cross_correlation /(summation_of_main_descriptors*summation_of_tempelte_descriptors)
            """
            if the ncc value is big it means that the features are the same
            so we compare to get the biggest values os ncc 
            """
            if float(ncc) > distance:
                distance = ncc
                matched_key_point = KeyPoint2

        cur = cv2.DMatch()
        cur.queryIdx = KeyPoint1
        cur.trainIdx = matched_key_point
        cur.distance = distance
        matches.append(cur)
    return matches

def NCC_MainFn(img):
    img1 = cv2.imread(img)
    img2 = cv2.imread("Images/rotated_deer.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    start = time.time()
    
    NCC_Matches = Normalized_Cross_Correlation(descriptors_1, descriptors_2)
    end = time.time()
    total_time = end-start
    print("NCC Execution Time=", total_time)

    NCC_Matches = sorted(NCC_Matches, key=lambda x: x.distance, reverse=True)
    NCC_Matched_Image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,NCC_Matches[:21], img2, flags=2)

    plot_NCC_image(img1=NCC_Matched_Image)
def SDD_MainFn(img):
    img1 = cv2.imread(img)
    img2 = cv2.imread("Images/rotated_deer.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    start = time.time()
    SSD_Matches = Sum_Square_Difference(descriptors_1, descriptors_2)
    end = time.time()
    total_time = end-start
    print("SSD Execution Time=", total_time)
    SSD_Matches = sorted(SSD_Matches, key=lambda x: x.distance, reverse=True)

    SSD_Matched_Image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,SSD_Matches[:21], img2, flags=2)
    plot_SDD_image(SSD_Matched_Image)

if __name__ == "__main__":
    pass
