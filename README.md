## Vessel Detection in Synthetic Aperture Radar(SAR) Images
This repo is created to evaluate the vessel detections in SAR images though traditional methods e.g different variants of CFAR and deep learning target detection architectures. 
An annotated dataset by SAR experts was recently published consisting of 43,819 ship chips is used to evaluate vessel detection "A SAR Dataset of Ship Detection for Deep Learning under Complex Backgrounds" [GitHub](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)
[Paper](https://www.mdpi.com/2072-4292/11/7/765/htm). This dataset is used to evaluate the detection. We split the dataset into training and evaluation sets. Evaluation set consists of Last 3819 images. First 40000 images are used for training deep-learning models. 

This repo detects vessels through CA CFAR and saves the results for further evaluation.
### Pre Processing

Morphological operations such as erosion is applied to the images. Erosion removes islands and small objects(speckle) so that only substantive targets remain in SAR images. Then we apply median blur filter to eroded image.

### Detection
We have implemented Cell Averaging Constant False Alarm Rate (CA-CFAR)detector to detect the targets(vessel). 
It compares the pixels or group of pixels to a threshold. Setting the threshold values determines the probability of false alarm and probability of detection.
To come up with the threshold value CA CFAR algorithm is used. A sliding window of RxC pixels is convolved over the image. The center pixel is cell under test (CUT), it can be single pixel or block of pixels. Pixels around CUT are left out of computation called guard cells. Cells around guard cells are background cells or training cells. Threshold value for each CUT is estimated through background cells by averaging the background cells (CA CFAR).
- T = a*Pn
    - T is estimated threshold
    - a is scaling factor
    - Pn is noise power estimated by background cells 
- CUT is classified as target(vessel) when CUT > T

CFAR algorithm and other pipeline operations are extended in python from c++ to gain real time detection results. 
We use opencv for Morphological operations, image blur filters, bounding boxes rendering. 

The detection boxes are drawn with ground truth boxes and saved for visualization, results are stored in txt file for each image with original filename in following format.

    Ship 0.8 x y w h

To run the detection on dataset set data_dir in run.py and run parse_xml.py to parse ground truth from xml file to a text file in following format
    
    Ship x y w h

    python parse_xml.py
    python setup.py install
    python run.py

