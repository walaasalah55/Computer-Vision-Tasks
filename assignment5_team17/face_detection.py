import cv2
from cv2 import CascadeClassifier

def face_detection(photoPath):
    # Create the haar cascade
    faceCascade = CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # Read the image
    resultImage = cv2.imread(photoPath)
    gray = cv2.cvtColor(resultImage, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(resultImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    facesNum = len(faces)

    return facesNum, resultImage     

if __name__ == '__main__':
    path='faces.jpeg'
    facesNum, resultImage = face_detection(path)
    fileName = 'resultImg.jpg'
    cv2.imwrite(fileName, resultImage)
    print("there are ", facesNum, "in the photo")
    cv2.imshow("Detected Faces", resultImage)
    cv2.waitKey(0)