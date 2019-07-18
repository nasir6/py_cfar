## Vessel Detection in Synthetic Aperture Radar(SAR) Images
### Dataset
A recently published dataset consisting of 43,819 ship chips is used to evaluate vessel detection "A SAR Dataset of Ship Detection for Deep Learning under Complex Backgrounds" [GitHub](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)
[Paper](https://www.mdpi.com/2072-4292/11/7/765/htm)

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

For more details and experimental results [DropBox Paper](https://paper.dropbox.com/doc/SAR-vessel-detection--Ag8sKJlxfjm1uQAg_B7BwnabAg-i6ifPVu9dKsqu7dwgKoJa)
