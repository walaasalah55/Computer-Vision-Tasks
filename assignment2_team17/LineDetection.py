import numpy as np
import cv2
import matplotlib.pyplot as plt
from utilis import plot_image,Canny_edge_detector



# This is the function that will build the Hough Accumulator for the given image
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape 
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    Hough_Accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) 

    for i in range(len(x_idxs)): 
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): 
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            Hough_Accumulator[rho, j] += 1

    return Hough_Accumulator, rhos, thetas


def hough_peaks(Hough_Accumulator, num_peaks, nhood_size=3):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to a local maxima.  If threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of indicies around the maximum will be surpessed. '''

    indicies = []
    Hough_Acc = np.copy(Hough_Accumulator)
    for i in range(num_peaks):
        idx = np.argmax(Hough_Acc) 
        Hough_Acc_idx = np.unravel_index(idx, Hough_Acc.shape) # remap to shape of H
        indicies.append(Hough_Acc_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = Hough_Acc_idx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > Hough_Accumulator.shape[1]): max_x = Hough_Accumulator.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > Hough_Accumulator.shape[0]): max_y = Hough_Accumulator.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                # remove neighborhoods in H1
                Hough_Acc[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    Hough_Accumulator[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    Hough_Accumulator[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, Hough_Accumulator


# drawing the lines from the Hough Accumulatorlines using OpevCV cv2.line
def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 500*(-b))
        y1 = int(y0 + 500*(a))
        x2 = int(x0 - 500*(-b))
        y2 = int(y0 - 500*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

'''
Original_Image=cv2.imread('images/chess.jpeg')
canny_edges=Canny_edge_detector('images/chess.jpeg')
Hough_Accumulator, rhos, thetas = hough_lines_acc(canny_edges)
indicies, Hough_Accumulator = hough_peaks(Hough_Accumulator, 18, nhood_size=11) # find peaks
hough_lines_draw(Original_Image, indicies, rhos, thetas)

# Show image with manual Hough Transform Lines
plot_image(img=Original_Image)
'''





