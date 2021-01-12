# DNN-Object-Detection-YOLO3
Deep learning based object detection using YOLOv3 with OpenCV

## There are three main object detectors using deep learning:-

1. R-CNN (Selective Search), Fast R-CNN( Region proposed Network and R-CNN).  
2. Single shot detectors (SSD).  
3. YOLO.  

Both SSD and YOLO use one-stage detector strategy.  

## Requirements:-  

1. OpenCV 4.5.0 and above.
2. Cuda 11.1 and above.
3. cudnn 8.0.5 and above.

## Usage:-  

### 1. Download the models
    $ sudo chmod +x models.sh
    $ ./models.sh

### 2. Build the project with cmake
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
### 3. Run
    $ ./opencv --video=PATH_TO_THE_VIDEO
