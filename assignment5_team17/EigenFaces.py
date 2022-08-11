import cv2
import os
from natsort import natsorted
import numpy as np
from numpy.linalg import eig
# from glob import glob

NumberOfImages = 360
NumberOfPixels = 92 * 112
image_shape = (92, 112) 
a = None
Images=[]
# Images = natsorted(glob('Training_Dataset' + '/**/*.pgm', recursive = True))
def Existed_Images():
    """
    This function get all images located in given path and iterate
    on them to flatten all 2D images to 1D vector ,then put
    All these 1D vectors in one array  
    """
    # create array to append the flattend images on it
    Matrix = np.zeros((NumberOfImages, NumberOfPixels, 1))

    for i in range(1, 41):
        folder = "s" + str(i)
        path_to_image = 'Training_Dataset/' + str(folder)
        images = [pos_image for pos_image in os.listdir(path_to_image) if pos_image.endswith('.pgm')]
        Sorted_List = natsorted(images)
        
        for image_number in range(1, 10):

            image_no = Sorted_List[image_number-1]
            path = path_to_image+'/'+str(image_no)
            Images.append(path)

            # Read the images
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = np.array(image)

            # convert the images from 2D array to Vetor 1D
            image_1D = image.flatten()
        
            if i <= 36:
                index = ( (i - 1) * 10) + (image_number - 1)

            # put the 1D vector in the big array
                Matrix[ index ] = np.vstack(image_1D)

    Matrix = np.resize(Matrix, (NumberOfImages, NumberOfPixels))
    return Matrix.T

def calculate_covariance(main_matrix):
    # calculate the mean
    mean_image = []
    rows = main_matrix.shape[0]
    for row in range(rows):
        mean_image.append(np.mean(main_matrix[row]))

    mean_image = np.reshape(mean_image, newshape=(10304, 1))

    # Substract the mean from the Matrix of all images
    a = main_matrix - mean_image
    print(np.shape(a))
    # Calculate the covariance matrix
    Covariance_Matrix = ( 1/NumberOfImages ) * ( np.dot( a.T, a ) )
    print(np.shape(Covariance_Matrix))

    return Covariance_Matrix, mean_image , a

def calculate_eigenfaces(Covariance_Matrix):
    eigenvalues, eigenvectors = eig(Covariance_Matrix)
    real_eigenval = eigenvalues.real
    real_eigenvect = eigenvectors.real
    total_sum_of_eigenvalues = np.sum(eigenvalues)
    sum_of_eigenvalues = 0
    counter = 0 
    for eigval in eigenvalues:
        if sum_of_eigenvalues/total_sum_of_eigenvalues < 0.9:
            sum_of_eigenvalues = sum_of_eigenvalues + eigval
            counter = counter + 1

    eigenfaces = real_eigenvect[:counter, :]

    return eigenfaces , counter


def proj_test_img(path, eigenfaces , mean_img, counter):
    # Testing image
    test_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # # resize the testing image. cv2 resize by width and height.
    test_img = cv2.resize(test_img, (image_shape[1] , image_shape[0]))

    # subtract the mean
    # test_img = np.reshape(test_img, (test_img.shape[0] * test_img.shape[1]))
    test_minus_mean =  np.reshape(test_img, (test_img.shape[0] * test_img.shape[1], 1)) - mean_img


    # the vector that represents the image with respect to the eigenfaces.
    projected_test_img = np.dot(eigenfaces.T, test_minus_mean[:counter, :])
    print(np.shape(projected_test_img))
    return projected_test_img.T

def calculate_similarity(eigenfaces, projected_img, a, counter):

     
    smallest_value = None       # to keep track of the smallest value
    index = 0                # to keep track of the class that produces the smallest value
    prop_list = []
    image_index=0
      
    for image in Images:
        # calculate the vectors of the images in the dataset and represent
        train_img = eigenfaces.T.dot(a[:counter,image_index])

        euclidean_distance = np.linalg.norm(projected_img - train_img)
        # best_match = np.argmin(euclidean_distance)

        if smallest_value is None:
            smallest_value = euclidean_distance
            index = image

        if smallest_value > euclidean_distance:
            smallest_value = euclidean_distance
            index = image
        
        prop_list.append([image_index, euclidean_distance])
        image_index=image_index+1
    return index, smallest_value , prop_list
