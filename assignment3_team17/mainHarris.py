import matplotlib.image as mpimg
from harris import * 

def mainFn(imgPath):
    # img = mpimg.imread( 'Harris-CV/Harris/Images/chess.jpeg' ) 
    # img = mpimg.imread( 'Harris-CV/Harris/Images/pexels.jpeg' )
    img = mpimg.imread(imgPath)
    imggray = from_RGB_to_GS( img )

    # apply Harris Corner Detection
    # k : Sensitivity factor
    # chess k = 0.1
    # pexels = 0.04
    #  15.jpg = 0.04
    harris_response = f_harris( imggray, k = 0.1)

    # categorize Harris response Edge, Corner, Flat 
    # chess threshold = 0.5 
    # pexels.jpeg threshold = 0.1
    # 15.jpg  = 0.1
    corners, edges = categorize_harris_response( img, harris_response , threshold = 0.5)

    plot_image(corners)