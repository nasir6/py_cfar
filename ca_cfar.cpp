
#include <stdlib.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace std;

int getoffset(int i, int window_size, int limit) {
  int min_index = i - floor(window_size/2);
  int max_index = i + floor(window_size/2);
  if (min_index > 0 && max_index < limit) {
    return 0;
  } else {
    if (min_index < 0) {
      return 0 -  min_index;
    } else {
      return limit - max_index - 1;
    }
  }
}
void CA_CFAR(cv::Mat& inputImage, cv::Mat& outputImage, int backgroundSize, int guardSize, int pixel_size, double thresholdValue) {
  outputImage = inputImage.clone();
  outputImage.setTo(cv::Scalar::zeros());
  
  int rows = inputImage.rows;
  int cols = inputImage.cols;

  int pixel = 0;
  int padSize = 0;//floor(backgroundSize / 2);
  // int padSize = floor(backgroundSize / 2);

  // int pixel_size = 20;
  /*
    TODO: add padding to input image
    Run through buffer while simulaneously filling the OpenCV matrix/image (raster).
  */

  for (int i = padSize; i < rows - padSize; i++) {
    for (int j = padSize; j < cols - padSize; j++) {
      double sum = 0.0, avg = 0.0, pixel_sum = 0.0;
      pixel = (int) inputImage.at<uchar>(i,j);
      if(pixel > 100) {
        int total_cut_pixels = 0;
        int total_bg_pixels = 0;
        if (pixel_size < 2) {
          pixel_sum = (int) inputImage.at<uchar>(i,j);
          total_cut_pixels = 1;
        } else {
          for(int x = -floor(pixel_size/2); x <= floor(pixel_size/2); x++) {
            for(int y = -floor(pixel_size/2); y <= floor(pixel_size/2); y++) {
              int r = i+y;
              int c = j+x;
              if (r < 0 || c < 0 || r >= rows || c >= cols){
                pixel_sum += 0;
                // total_cut_pixels +=1;
              } else {
                total_cut_pixels +=1;
                pixel_sum += (int) inputImage.at<uchar>(r, c);
              }
            }
          }
        }
        int offsetX = getoffset(i, backgroundSize, rows);
        int offsetY = getoffset(j, backgroundSize, cols);

        for(int x = -floor(backgroundSize/2); x <= floor(backgroundSize/2); x++) {
          for(int y = -floor(backgroundSize/2); y <= floor(backgroundSize/2); y++) {
            int r = i+y+offsetX;
            int c = j+x+offsetY;
            if (r < 0 || c < 0 || r >= rows || c >= cols){
              sum += 0;
            } else {
              sum += (int) inputImage.at<uchar>(r, c);
              total_bg_pixels+=1;
            }
          }
        }
        offsetX = getoffset(i, guardSize, rows);
        offsetY = getoffset(j, guardSize, cols);
        for(int x = -floor(guardSize/2); x <= floor(guardSize/2); x++) {
          for(int y = -floor(guardSize/2); y <= floor(guardSize/2); y++) {
            int r = i+y+offsetX;
            int c = j+x+offsetY;
            if (r < 0 || c < 0 || r >= rows || c >= cols) {
              sum -=0;
            }else{
              sum -= (int) inputImage.at<uchar>(r, c);
              total_bg_pixels-=1;

            } 
          }
        }
        
        avg = sum/total_bg_pixels;
        double pixel_avg = pixel_sum / total_cut_pixels;
        if (pixel_avg > thresholdValue*avg) {
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

