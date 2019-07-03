#ifndef CA_CFAR_H
#define CA_CFAR_H


#include <stdlib.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace std;


class CA_CFAR {
    int rows = 0;
    int cols = 0;
    int pixel = 0;
    int offsetX = 0;
    int offsetY = 0;
    double cut_avg = 0;
    double bg_avg = 0.0;
    
    int backgroundSize;
    int guardSize;
    int pixel_size;
    double thresholdValue;

    vector<double> cut_sum;
    vector<double> guard_sum;
    vector<double> bg_sum;

    int getEdgeOffset(int i, int window_size, int limit);
    vector<double> get_block_sum(cv::Mat& inputImage, int i, int j, int blockSize);

    public:
        CA_CFAR(int _backgroundSize, int _guardSize, int _pixel_size, double _thresholdValue);
        void mask(cv::Mat& inputImage, cv::Mat& outputImage);
  };

#endif