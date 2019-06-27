#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Python.h>

#include <numpy/arrayobject.h>

Mat Erosion( Mat &src, Mat &erosion_dst, int erosion_size )
{
    int erosion_type = 0;
    int erosion_elem = 2;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement( erosion_type,
                        Size( erosion_size, erosion_size ));
    erode( src, erosion_dst, element );
    return erosion_dst;
}

PyObject * build_image_array(Mat &outputImage) {
    int rows = outputImage.rows;
    int cols = outputImage.cols;
    int nElem = cols * rows;
    uchar* m = new uchar[nElem];
    std::memcpy(m, outputImage.data, nElem * sizeof(uchar));
    npy_intp mdim[] = { rows, cols };
    PyObject* mat = PyArray_SimpleNewFromData(2, mdim, NPY_UINT8, (void*) m);
    return mat;
}

