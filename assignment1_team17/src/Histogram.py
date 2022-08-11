import numpy as np
from PIL import Image, ImageOps
import math

# Histogram function
def HistDistFun(GreyScaleImg):
    Frequencies=[]
    PixelValues=[]

    for s in range(0,256,1):
        PixelValues.append(s)
        
    for k in range(0,256,1):
        Frequencies.append(0)
        
    imgShape=GreyScaleImg.shape
    cols,rows=imgShape
    for i in range(0,cols,1):
        for j in range(0,rows,1):
            value=GreyScaleImg[i,j]         
            Frequencies[value]=Frequencies[value]+1        

    return PixelValues,Frequencies 
#################################################################
# RGB histograms and distribution curves
def RGBHistDistFun(RGBImage):
    BlueColorsFreq,GreenColorsFreq,RedColorsFreq=[],[],[]
    BlueColorsValues,GreenColorsValues,RedColorsValues=[],[],[]
    imgShape=RGBImage.shape
    cols,rows,_=imgShape
    for k in range(0,256,1):
        BlueColorsFreq.append(0)
        GreenColorsFreq.append(0)
        RedColorsFreq.append(0)
    for num in range(0,256,1):
        BlueColorsValues.append(num)
        GreenColorsValues.append(num)
        RedColorsValues.append(num)        
    for i in range(0,cols,1):
            for j in range(0,rows,1):
                blue=RGBImage[i,j][2]
                green=RGBImage[i,j][1]
                red=RGBImage[i,j][0]
                BlueColorsFreq[blue]=BlueColorsFreq[blue]+1
                GreenColorsFreq[green]=GreenColorsFreq[green]+1
                RedColorsFreq[red]=RedColorsFreq[red]+1
    return BlueColorsValues,GreenColorsValues,RedColorsValues,BlueColorsFreq,GreenColorsFreq,RedColorsFreq
###############################################################
# image equalization
# cumulative distribution frequency
def cdf(histogram):  
    cdf = [0] * len(histogram)   #len=256
    #print (cdf)
    cdf[0] = histogram[0]
    for i in range(1, len(histogram)):

        cdf[i]= cdf[i-1]+histogram[i]
    # normalization from 0 to 1
    cdf = [n*255/cdf[-1] for n in cdf]     

    return cdf

# to get new pixel values (linear interpolation) 
def image_equalization(image):
    _,Freqs=HistDistFun(image)
    img_cdf = cdf(Freqs)
    equalized_img = np.interp(image, range(0,256), img_cdf)

    return equalized_img
####################################################################

#Image normalization
def normalization(cols, rows, x_max, x_min, x_new_max, x_new_min,Img):

    #img = Image.open(Img)
    #img = img.resize((cols, rows), Image.ANTIALIAS)
    img_norm_list = []
    for i in range(0, cols):
        img_norm_list_rows = []
        for j in range(0, rows):
            #normalization formula is: x_norm = (x - x_new_min) / (x_max - x_min)) * (x_new_max - x_new_min)
            x_norm = ((Img[i][j] - x_new_min) / (x_max - x_min)) * (x_new_max - x_new_min)
            img_norm_list_rows.append(math.ceil(x_norm))
        img_norm_list.append(img_norm_list_rows)

    array = np.array(img_norm_list)
    h, w = array.shape
    mat = np.reshape(array, (h, w))
    img_norm = Image.fromarray(mat)

    return img_norm
######################################################################


