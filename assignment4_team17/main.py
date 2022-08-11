from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *   
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from os import path
import sys
import cv2
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import Segmentation.Kmean,Segmentation.regionGrowing,Segmentation.Agglomerative,Segmentation.luv,Segmentation.meanShift
import Thresholding.optimal_threshold,Thresholding.otsu,Thresholding.spectral,Thresholding.Local_Thresholding
MAIN_WINDOW,_=loadUiType(path.join(path.dirname(__file__),"gui.ui"))

class MainApp(QMainWindow,MAIN_WINDOW):
  
    def __init__(self,parent=None):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.menuBar()
        self.images={}
        self.img_comboBox=[self.comboBox]
        self.comboBox.setCurrentText('Colored Images')
        self.comboBox_2.setCurrentText('Grey Images')
   
    def menuBar(self):
        self.openFirstImg.triggered.connect(lambda:self.browse())
    
    def browse(self):
        self.fileName = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","*")
        pixmap = QPixmap(self.fileName[0])
        self.label_2.setPixmap(pixmap)
        self.label_2.setScaledContents(True)
        self.img_comboBox[0].currentIndexChanged.connect(lambda:self.ComboBox_function(self.fileName[0]))
        self.comboBox_2.currentIndexChanged.connect(lambda:self.ComboBox_function(self.fileName[0]))
    
    def ComboBox_function(self,img):
            if self.comboBox.currentText()=='RGB to LUV':
                Segmentation.luv.get_LUV_output(img)
                pixmap2= QPixmap('LUV_output.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox_2.currentText()=='Optimal Thresholding':
                Thresholding.optimal_threshold.get_optimalThreshold_output(img)
                pixmap2= QPixmap('optimal_threshold.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox.currentText()=='K-Mean Segmentation':
                Segmentation.Kmean.get_kmean_output(img)
                pixmap2= QPixmap('Kmean.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox.currentText()=='Region Growing Segmentation':
                Segmentation.regionGrowing.get_regionGrowing_output(img)
                pixmap2= QPixmap('regionOutput.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox.currentText()=='Agglomerative Segmentation':
                Segmentation.Agglomerative.get_agglomerative_output(img)
                pixmap2= QPixmap('agglomerative.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox.currentText()=='Mean Shift Segmentation':
                Segmentation.meanShift.get_meanshift_output(img)
                pixmap2= QPixmap('meanShift.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox_2.currentText()=='Otsu Thresholding':
                Thresholding.otsu.get_otsu_output(img)
                pixmap2= QPixmap('otsu.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox_2.currentText()=='Spectral Thresholding':
                Thresholding.spectral.get_spectral_output(img)
                pixmap2= QPixmap('spectral.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox_2.currentText()=='Local Optimal Thresholding':
                Thresholding.Local_Thresholding.get_Local_Threshold_output(img,2,2,'optimal_threshold')
                pixmap2= QPixmap('Local.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox_2.currentText()=='Local Otsu Thresholding':
                Thresholding.Local_Thresholding.get_Local_Threshold_output(img,2,2,'otsu_threshold')
                pixmap2= QPixmap('Local.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()