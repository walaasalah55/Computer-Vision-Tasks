import numpy as np
import cv2


# with np.errstate(divide='ignore', invalid='ignore'):  # to ignore divide by 0 warning 


def RGB2LUV(image: np.ndarray):

    RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB image to XYZ
    image = RGBimage / 255.0
    X = np.dot(image, [0.412453, 0.357580, 0.180423])
    Y = np.dot(image, [0.212671, 0.715160, 0.072169])
    Z = np.dot(image, [0.019334, 0.119193, 0.950227])
    #print(Y.shape)

    # Convert XYZ image to LUV
    Un = 0.19793943
    Vn = 0.46831096
    u_dash = np.divide((4.0 * X), (X + (15.0 * Y) + (3.0 * Z)))
    v_dash = np.divide((9.0 * Y), (X + (15.0 * Y) + (3.0 * Z)))

    L = np.where(Y > 0.008856, 116 * np.power(Y, 1 / 3) - 16, 903.3 * Y)
    U = 13*L*(u_dash - Un)
    V = 13*L*(v_dash - Vn)
    
    # scaling values for 8-bit images
    L = (255.0 / 100)*L
    U = (255.0 / 354)*(U + 134)
    V = (255.0 / 262)*(V + 140)
    LUV = np.dstack((L, U, V)).astype(np.uint8)

    return LUV

def get_LUV_output(path):
    Image = cv2.imread(path)  
    LUV=RGB2LUV(Image)
    #print(LUV)
    #cv2.imshow('image',Image)
    # cv2.imshow('LUV', LUV)
    # with cv
    cv_luv = cv2.cvtColor(Image, cv2.COLOR_BGR2LUV)
    convertedImg = cv2.convertScaleAbs(cv_luv, alpha=(255.0))
    cv2.imwrite('LUV_output.jpg',cv_luv)