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
from ImageMatching import NCC_MainFn, SDD_MainFn
from mainHarris import mainFn as HarrisMain
MAIN_WINDOW,_=loadUiType(path.join(path.dirname(__file__),"gui.ui"))

class MainApp(QMainWindow,MAIN_WINDOW):
  
    def __init__(self,parent=None):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.menuBar()
        self.images={}
        self.img_comboBox=[self.comboBox]
   
    def menuBar(self):
        self.openFirstImg.triggered.connect(lambda:self.browse())
    
    def browse(self):
        self.fileName = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","*")
        pixmap = QPixmap(self.fileName[0])
        self.label_2.setPixmap(pixmap)
        self.label_2.setScaledContents(True)
        self.img_comboBox[0].currentIndexChanged.connect(lambda:self.ComboBox_function(self.fileName[0]))
    
    def ComboBox_function(self,img):
            if self.comboBox.currentText()=='Harris':
                HarrisMain(img)
                pixmap2= QPixmap('Harris_Image.jpg')
                self.label_3.setPixmap(pixmap2)
                self.label_3.setScaledContents(True)
            elif self.comboBox.currentText()=='Sift':
                img = cv2.imread(img)
                gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                sift = cv2.SIFT_create()
                kp = sift.detect(gray,None)
                img=cv2.drawKeypoints(gray,kp,img)
                cv2.imwrite('sift_keypoints.jpg',img)
                pixmap = QPixmap('sift_keypoints.jpg')
                self.label_3.setPixmap(pixmap)
                self.label_3.setScaledContents(True)
            elif self.comboBox.currentText()=='Image Matching SSD':
                SDD_MainFn(img)
                pixmap = QPixmap('SDD_Matched_Image.jpg')
                self.label_3.setPixmap(pixmap)
                self.label_3.setScaledContents(True)
            elif self.comboBox.currentText()=='Image Matching NCC':
                NCC_MainFn(img)
                pixmap = QPixmap('NCC_Matched_Image.jpg')
                self.label_3.setPixmap(pixmap)
                self.label_3.setScaledContents(True)
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()