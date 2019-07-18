
#include <stdlib.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "ca_cfar.h"

using namespace std;


int CA_CFAR::getEdgeOffset(int i, int window_size, int limit) {
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

vector<double> CA_CFAR::get_block_sum(cv::Mat& inputImage, int i, int j, int blockSize) {

  int totalPixels = 0;
  int blockSum = 0;
  for(int x = -floor(blockSize/2); x <= floor(blockSize/2); x++) {
    for(int y = -floor(blockSize/2); y <= floor(blockSize/2); y++) {
      int r = i+y+offsetX;
      int c = j+x+offsetY;
      if (r < 0 || c < 0 || r >= rows || c >= cols){
        continue;
      } else {
        totalPixels +=1;
        blockSum += (int) inputImage.at<uchar>(r, c);
      }
    }
  }
  vector<double> sum_count = {(double) blockSum ,(double) totalPixels}; 
  return sum_count;
}

CA_CFAR::CA_CFAR(int _backgroundSize, int _guardSize, int _pixel_size, double _thresholdValue) {
  backgroundSize = _backgroundSize;
  guardSize = _guardSize;
  pixel_size = _pixel_size;
  thresholdValue = _thresholdValue;
}

void CA_CFAR::mask(cv::Mat& inputImage, cv::Mat& outputImage) {
  outputImage = inputImage.clone();
  outputImage.setTo(cv::Scalar::zeros());
  
  rows = inputImage.rows;
  cols = inputImage.cols;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      pixel = (int) inputImage.at<uchar>(i,j);
      if(pixel < MINIMUM_PIXEL_VALUE) {
        outputImage.at<uchar>(i,j) = 0;
        continue;
      }

      if (pixel_size < 2) {
        cut_avg = (int) inputImage.at<uchar>(i,j);
      } else {
        offsetX = 0;
        offsetY = 0;
        cut_sum = get_block_sum(inputImage, i, j, pixel_size);
        cut_avg = cut_sum[0] / cut_sum[1];
      }

      offsetX = getEdgeOffset(i, backgroundSize, rows);
      offsetY = getEdgeOffset(j, backgroundSize, cols);
      bg_sum = get_block_sum(inputImage, i, j, backgroundSize);
      
      offsetX = getEdgeOffset(i, guardSize, rows);
      offsetY = getEdgeOffset(j, guardSize, cols);
      guard_sum = get_block_sum(inputImage, i, j, guardSize);

      bg_avg = (bg_sum[0] - guard_sum[0])/(bg_sum[1] - guard_sum[1]);
      
      if (cut_avg > thresholdValue*bg_avg) {
        outputImage.at<uchar>(i,j) = 255;
      } else {
        outputImage.at<uchar>(i,j) = 0;
      }
    }
  }
}

