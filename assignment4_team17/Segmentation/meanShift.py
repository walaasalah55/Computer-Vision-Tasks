import numpy as np
from matplotlib import pyplot as plt
import cv2


class meanShiftSeg:

    def __init__(self, image, windowsize):
        self.image = np.array( image, copy = True )       
        self.segmentedImage = np.array( image, copy = True )
        self.colorSpace = np.zeros( (256,256) )
        self.windowsize = 2**(windowsize) 
        self.clusters_num = int(256/self.windowsize)**2
        self.clustersUV = np.zeros( shape= (self.clusters_num, 2) )       
   
    # get uv ,apply mean shift to  colorspace and classify the UV components
    def applyMeanShift(self):

        Ucomp = np.reshape( self.image[:,:,1], (-1,1) )
        Vcomp = np.reshape( self.image[:,:,2], (-1,1) )
        UVcomp = np.transpose(np.array((Ucomp[:,0],Vcomp[:,0])))

        for u,v in UVcomp :
            self.colorSpace[ u,v ] =self.colorSpace[ u,v ]+ 1
        
        numOfWindPerDim = int(np.sqrt( self.clusters_num ))
        clustersTemp = []
        for itrRow in range( numOfWindPerDim ):
            for itrCol in range( numOfWindPerDim ):
                cntrRow, cntrCol = self.windowIter(int(itrRow*self.windowsize),int(itrCol*self.windowsize)) 
                clustersTemp.append( (cntrRow, cntrCol) )

        self.clustersUV = np.array( clustersTemp )
        #print (self.clustersUV)  #centers
        self.classifycomponent()

        return self.segmentedImage

   
    def windowIter(self, row, col):
       
        # iterate in the given window indices, to find its center of mass

        hWSize = self.windowsize/2
        prevRow = 0
        prevCol = 0
       
        window = self.colorSpace[ row:row+self.windowsize,col:col+self.windowsize ]        
        newRow, newCol = self.findCenterMass( window )
        numOfIter = 0
        while( prevRow != newRow-hWSize and prevCol != newCol-hWSize ):
            if( numOfIter > np.sqrt(self.clusters_num) ):
                break

            prevRow = newCol-hWSize
            prevCol = newCol-hWSize

            nxtRow = int((prevRow+row)%(256-self.windowsize))
            nxtCol = int((prevCol+col)%(256-self.windowsize))
            window = self.colorSpace[ nxtRow:nxtRow+self.windowsize,nxtCol:nxtCol+self.windowsize ]
            newRow, newCol = self.findCenterMass( window )
            numOfIter += 1
        return (row + newRow) , (col + newCol)

    def classifycomponent(self):

            wSize = self.windowsize
            numOfWindPerDim = int(np.sqrt( self.clusters_num ))
            for row in range( self.image.shape[0] ):
                for col in range( self.image.shape[1] ):
                    pixelU = self.segmentedImage[row,col,1]
                    pixelV = self.segmentedImage[row,col,2]
                    windowIdx = int( int(pixelV/wSize)  + int(numOfWindPerDim*( pixelU/wSize )))
                    self.segmentedImage[row,col,1] = self.clustersUV[windowIdx, 0]
                    self.segmentedImage[row,col,2] = self.clustersUV[windowIdx, 1]

    def findCenterMass(self, window):

        momntIdx = range( self.windowsize )
        totalMass = np.max(np.cumsum( window ))
        if (totalMass == 0):
            return self.windowsize/2 , self.windowsize/2
        if ( totalMass > 0 ):
            # around the x-axis  col 0
            momentCol = np.max(np.cumsum(window.cumsum( axis=0 )[self.windowsize-1]*momntIdx))
            cntrCol = np.round(1.0*momentCol/totalMass)
            # around the y-axis  row 0
            momentRow = np.max(np.cumsum(window.cumsum( axis=1 )[:,self.windowsize-1]*momntIdx))
            cntrRow = np.round(1.0*momentRow/totalMass)

            return cntrRow, cntrCol

    def findEclidDist(self, row, col):
        return (np.round(np.sqrt( (row**2 + col**2 ))))

    def getSegmentedImage(self):
        return self.segmentedImage



def get_meanshift_output(path_to_jpg_file):

    RGBimage = cv2.imread(path_to_jpg_file )
    imageLUV = cv2.cvtColor( RGBimage, cv2.COLOR_RGB2LUV )

    meanShift = meanShiftSeg( imageLUV, 7 ) # image,windowsize
    segImage = meanShift.applyMeanShift()

    # cv2.imshow( 'image', segImage )
    cv2.imwrite('meanShift.jpg',segImage)
# get_meanshift_output( "Images/Agglomerative.jpeg ")

