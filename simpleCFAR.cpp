
#include <stdlib.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace std;

void CA_CFAR(cv::Mat& inputImage, cv::Mat& outputImage, int backgroundSize, int guardSize, double thresholdValue) {
  outputImage = inputImage.clone();
  // outputImage.create(inputImage.rows,inputImage.cols, inputImage.type());
  outputImage.setTo(cv::Scalar::zeros());
  
  int rows = inputImage.rows;
  int cols = inputImage.cols;

  int pixel = 0;
  // int backgroundSize = 10;
  int padSize = 0;//floor(backgroundSize / 2);
  // int guardSize = 3;
  // double thresholdValue = 0.9;
  
  /*
    TODO: add padding to input image
   */
  /// Run through buffer while simulaneously filling the OpenCV matrix/image (raster).
  for (int i = padSize; i < rows - padSize; i++) {
    for (int j = padSize; j < cols - padSize; j++) {
      double sum = 0.0, avg = 0.0;
      pixel = (int) inputImage.at<uchar>(i,j);
      if(pixel > 100) {
        for(int x = -floor(backgroundSize/2); x <= floor(backgroundSize/2); x++) {
          for(int y = -floor(backgroundSize/2); y <= floor(backgroundSize/2); y++) {
            int r = i+y;
            int c = j+x;
            if (r < 0 || c < 0){
              sum += 0;
            } else {
              sum += (int) inputImage.at<uchar>(r, c);
            }
          }
        }

        for(int x = -floor(guardSize/2); x <= floor(guardSize/2); x++) {
          for(int y = -floor(guardSize/2); y <= floor(guardSize/2); y++) {
            int r = i+y;
            int c = j+x;
            if (r < 0 || c < 0){
              sum -=0;
            }else{
              sum -= (int) inputImage.at<uchar>(r, c);
            } 
          }
        }
        
        avg = sum/(backgroundSize*backgroundSize - guardSize*guardSize);
  
        if (pixel > thresholdValue*avg) {
          outputImage.at<uchar>(i,j) = 255;

        } else {
          outputImage.at<uchar>(i,j) = 0;
        }

      } else {
        outputImage.at<uchar>(i,j) = 0;
      }
    }
  }
}


