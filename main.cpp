#include <Python.h>

#include <numpy/arrayobject.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <simpleCFAR.cpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

static char cfar_docs[] =
   "cfar extension";

typedef struct {
    PyObject_HEAD
} PyCfar;

static void PyCfar_dealloc(PyCfar * self)
{
    Py_TYPE(self)->tp_free(self);
}


static PyObject *
PyCfar_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyCfar *self;
    self = (PyCfar *) type->tp_alloc(type, 0);
    
    return (PyObject *) self;
}

PyObject * cfar_init(PyCfar *self, PyObject *args, PyObject *kwds)
{
    return Py_BuildValue("i", 0);
}

void save_boxes(vector<Rect> boundRect,char * output_file) {
    ofstream myfile;
    myfile.open(output_file);
    for (size_t idx = 0; idx < boundRect.size(); idx++) {
        myfile << "Ship 0.8 ";
        myfile << boundRect[idx].tl().x;
        myfile << " ";
        myfile << boundRect[idx].tl().y;
        myfile << " ";
        myfile << boundRect[idx].br().x - boundRect[idx].tl().x;
        myfile << " ";
        myfile << boundRect[idx].br().y - boundRect[idx].tl().y;
        myfile << "\n";
    }
    myfile.close();
    return;
}

void draw_boxes(Mat &image, Mat &color, char * boxes_file) {

    std::vector<std::vector<cv::Point> > contours;
    Mat contourOutput;
    contourOutput = image.clone();
    
    cv::findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
    std::vector<Rect> boundRect( contours.size() );
    vector<vector<Point> > contours_poly( contours.size() );

    

    for (size_t idx = 0; idx < contours.size(); idx++) {
        
        cv::approxPolyDP( Mat(contours[idx]), contours_poly[idx], 3, true );
        boundRect[idx] = boundingRect( Mat(contours_poly[idx]) );
        if(cv::contourArea(contours[idx]) > 20) {
            rectangle( color, boundRect[idx].tl(), boundRect[idx].br(), cv::Scalar(0, 0, 255), 2, 8, 0 );
        }
    }
    save_boxes(boundRect, boxes_file);
    return;
}
void save_image(Mat &gray, Mat &im2, char * output_file, char * gt_file) {

    Mat im1(Size(gray.cols, gray.rows), CV_8UC3, Scalar(255,255,255));

    cv::cvtColor(gray, im1, cv::COLOR_GRAY2BGR); 

    int x, y, w, h;
    char* c_name;
    std::ifstream file(gt_file);
    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            // using printf() in all tests for consistency
            std::stringstream ss(line);
            std::string buf;
            int flag = 0;
            while (ss >> buf){
                if (flag == 1) {
                    x = std::stoi(buf);
                } else if (flag == 2) {
                    y = std::stoi(buf);
                } else if (flag == 3) {
                    w = std::stoi(buf);
                } else if (flag == 4) {
                    h = std::stoi(buf);
                }
                flag+=1;
            }
            rectangle( im1, Point(x, y), Point(w+x, y+h), cv::Scalar(0, 255, 0), 2, 8, 0 );

        }
        file.close();
    }



    Mat matDst(Size(im1.cols*2 + 150, im1.rows+ + 100),im1.type(),Scalar(255,255,255));
    Mat matRoi = matDst(Rect(50,50,im2.cols,im2.rows));
    im1.copyTo(matRoi);
    matRoi = matDst(Rect(im1.cols + 100 ,50,im1.cols,im1.rows));
    im2.copyTo(matRoi);
    imwrite( output_file, matDst );

}
Mat Erosion( Mat &src, Mat &erosion_dst, int erosion_size )
{
    int erosion_type = 0;
    int erosion_elem = 1;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement( erosion_type,
                        Size( erosion_size+1, erosion_size+1 ));
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

static PyObject * PyCfar_execute(PyCfar *self, PyObject* args)
{
    char *input_file;
    char *output_file;
    char *boxes_file;
    char *gt_file;
    int background = 10;
    int guard = 4;
    double threshhold = 0.9;
    if (!PyArg_ParseTuple(args, "s|s|s|s|i|i|d|d", &input_file, &output_file, &boxes_file, &gt_file, &background, &guard, &threshhold)) {
        return Py_BuildValue("s","qqq");;
    }    
    Mat inputImage;
    inputImage = imread(input_file);   // Read the file

    if(! inputImage.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return Py_BuildValue("s","www");;
    }
    
    Mat gray_image;

    const int channels = inputImage.channels();
    if (channels > 1) {
        cvtColor(inputImage, gray_image, COLOR_BGR2GRAY);
    } else {
        gray_image = inputImage.clone();
    }

    Mat outputImage;
    Mat inputErodedImage;
    inputErodedImage = gray_image.clone();
    // Erosion(gray_image, inputErodedImage, 2);

    CA_CFAR(inputErodedImage, outputImage, background, guard, threshhold);
    Mat eroded;
    eroded = outputImage.clone();
    // Erosion(outputImage, eroded, 0);
    Mat boxes_drawn(Size(eroded.cols, eroded.rows), CV_8UC3, Scalar(255,255,255));
    cv::cvtColor(eroded, boxes_drawn, cv::COLOR_GRAY2BGR);
    draw_boxes(eroded, boxes_drawn, boxes_file);
    save_image(gray_image, boxes_drawn, output_file, gt_file);
    
    return Py_BuildValue("O", build_image_array(boxes_drawn));
}


static PyMethodDef cfar_funcs[] = {
   {"execute_cfar", (PyCFunction)PyCfar_execute, 
      METH_VARARGS, cfar_docs},
    {"cfar_init", (PyCFunction)cfar_init,
        METH_VARARGS, cfar_docs
    },
      {NULL}
};

static PyModuleDef cfarmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "cfar",
    .m_doc = cfar_docs,
    .m_size = -1,
};


static PyTypeObject PyCfarType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name="cfar",                                  /*tp_name*/
    .tp_basicsize=sizeof(PyCfar),                                /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    .tp_dealloc=(destructor)PyCfar_dealloc,                 /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_reserved*/
    0,                                          /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    0,                                          /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    .tp_flags = Py_TPFLAGS_DEFAULT,                         /*tp_flags*/
    .tp_doc = cfar_docs,                                          /*tp_doc*/
    0,              /*tp_traverse*/
    0,                      /*tp_clear*/
    0,                                          /*tp_richcompare*/
    0,                                          /*tp_weaklistoffset*/
    0,                                          /*tp_iter*/
    0,                                          /*tp_iternext*/
    .tp_methods=cfar_funcs,                                          /*tp_methods*/
    0,                                /*tp_members*/
    0,                                          /*tp_getsets*/
    0,                                          /*tp_base*/
    0,                                          /*tp_dict*/
    0,                                          /*tp_descr_get*/
    0,                                          /*tp_descr_set*/
    0,                                          /*tp_dictoffset*/
    .tp_init = (initproc)cfar_init,                      /*tp_init*/
    0,                                          /*tp_alloc*/
    .tp_new=PyCfar_new,                                 /*tp_new*/
};


PyMODINIT_FUNC PyInit_cfar(void)
// create the module
{
    PyObject* m;

    if (PyType_Ready(&PyCfarType) < 0){

        return NULL;
    
    }
    import_array();

    m = PyModule_Create(&cfarmodule);
    if (m == NULL){
        return NULL;
    }

    Py_INCREF(&PyCfarType);
    PyModule_AddObject(m, "cfar", (PyObject *)&PyCfarType); // Add Voice object to the module
    
    return m;
}
