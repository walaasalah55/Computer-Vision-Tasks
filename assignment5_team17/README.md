
<!-- Task #5 Report -->

# <a name="face-detection-and-recognition_h">Face Detection and Recognition</a>
In this section we present Face Detection and Recognition implementations; `Using PCA/Eigenfaces Analysis`.

## 1. Face Detection
This is implemented using openCV library, using `CascadeClassifier` which contains OpenCV data used to detect objects.

There are mainly 2 parameter you can adjust in Face Detection:
- `Scale Factor`: Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.
- `Minimum Window Size`: size of each moving window that that algorithm to detect objects.


### Face Detection of one person
<!-- <img src="resources/results/face_detection_and_recognition/Face_Detection_1.png" alt="Face_Detection_1" width="600" height="500">

<img src="resources/results/face_detection_and_recognition/Face_Detection_3.png" alt="Face_Detection_3" width="600" height="500"> -->


## 2. Face Recognition
This implementation is based on `PCA/Eigenfaces Analysis`, it's implemented from scratch with the help of some useful libraries.

### Quick Description
- First you need to load the training dataset which consists of 40 class (folder), each class represents one person. Each class has 10 images, taken in different positions.
- Create Eigen-faces matrix for the dataset, which will be used later to compare with any new test image.
- Load your test image then start the matching (recognition) process.
<!-- - This function runs in a separate QThreads to ensure best quality and prevent GUI from freezing as it may take some seconds to finish.  -->

<!-- ### Face Recognition with correct test -->
<!-- <img src="resources/results/face_detection_and_recognition/Face_Recognition_1.png" alt="Face_Recognition_1" width="600" height="500">

<img src="resources/results/face_detection_and_recognition/Face_Recognition_2.png" alt="Face_Recognition_2" width="600" height="500"> -->

<!-- The output image `Best Match`, is just a combination of all the class images in the database, just for displaying purposes to make it clear to the user. -->
<!-- 
#### Face Recognition with wrong test
<img src="resources/results/face_detection_and_recognition/Face_Recognition_3.png" alt="Face_Recognition_3" width="600" height="500"> -->

<!-- <div style="page-break-after: always;"></div> -->


This repository is created by a group of 5 students in Biomedical Engineering Department, Cairo University. :copyright:


| Name                    | Section | B.N Number   |
|-------------------------|---------|--------------|
| Awatef Ahmed            | 2       |            6 |
| Kholud Abdelaazem       | 1       |           29 |
| Meirna Kamal            | 2       |           36 |
| Heba Elbeshbeshy        | 2       |           42 |
| Walaa Salah             | 2       |           45 |

