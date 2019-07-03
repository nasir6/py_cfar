#include <Python.h>

#include <numpy/arrayobject.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ca_cfar.cpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <plot_bbox.cpp>
#include <utils.cpp>

using namespace cv;
using namespace std;

static char cfar_docs[] =
   "runs cfar";

typedef struct {
    PyObject_HEAD
} PyCfar;


static PyObject *
PyCfar_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyCfar *self;
    self = (PyCfar *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}

static PyObject * PyRunCaCfar(PyCfar *self, PyObject* args) {
    char *input_file;
    char *output_file;
    char *boxes_file;
    char *gt_file;
    int background = 10;
    int guard = 4;
    int pixel_size = 4;
    double threshhold = 0.9;
    
    if (!PyArg_ParseTuple(args, "s|s|s|s|i|i|i|d|d", &input_file, &output_file, &boxes_file, &gt_file, &background, &guard, &pixel_size, &threshhold)) {
        return Py_BuildValue("s","Error parsing arguments");;
    }

    Mat inputImage;
    inputImage = imread(input_file);

    if(!inputImage.data) {
        cout <<  "Could not open or find the image" << std::endl ;
        return Py_BuildValue("s","could not read image file");;
    }
    
    Mat gray_image;
    const int channels = inputImage.channels();
    if (channels > 1) {
        cvtColor(inputImage, gray_image, COLOR_BGR2GRAY);
    } else {
        gray_image = inputImage.clone();
        cvtColor(gray_image, inputImage, COLOR_GRAY2BGR);
    }
    
    Mat inputErodedImage = gray_image.clone();
    Erosion(gray_image, inputErodedImage, 1);

    Mat filtered = gray_image.clone();

    // GaussianBlur( gray_image, filtered, Size( 5, 5 ), 0, 0 );
    medianBlur (inputErodedImage, filtered, 5 );
    // equalizeHist( gray_image, filtered );


    Mat outputImage;
    CA_CFAR ca_cfar(background, guard, pixel_size, threshhold);
    ca_cfar.mask(filtered, outputImage);
    
    Mat outputEroded;
    outputEroded = outputImage.clone();
    // Erosion(outputImage, outputEroded, 1);
    
    Mat boxes_drawn(Size(outputEroded.cols, outputEroded.rows), CV_8UC3, Scalar(255,255,255));
    cv::cvtColor(outputEroded, boxes_drawn, cv::COLOR_GRAY2BGR);

    // vector<Rect> boundRect = 
    vector<vector<int> > boundBoxes = find_boxes(outputEroded);
    save_boxes(boundBoxes, boxes_file);
    vector<vector<int> > gt_boxes = readGtBoxes(gt_file);

    draw_boxes(boundBoxes, boxes_drawn, cv::Scalar(0, 0, 255));
    draw_boxes(gt_boxes, inputImage, cv::Scalar(0, 255, 0));

    merge_save_image(inputImage, boxes_drawn, output_file);

    return Py_BuildValue("O", build_image_array(outputEroded));
}


static PyMethodDef cfar_funcs[] = {
   {"ca_cfar", (PyCFunction)PyRunCaCfar, 
      METH_VARARGS, cfar_docs},
      {NULL}
};

static PyModuleDef cfarmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "cfar",
    .m_doc = cfar_docs,
    .m_size = -1,
    cfar_funcs
};


PyMODINIT_FUNC PyInit_cfar(void)
{
    
    import_array();
    PyObject* m = PyModule_Create(&cfarmodule);
    if (m == NULL){
        return NULL;
    }
    return m;
}
