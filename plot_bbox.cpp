
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>


using namespace cv;
using namespace std;
const int MINIMUM_SHIP_SIDE = 15;

void save_boxes(vector<vector<int> > boundBoxes, char * output_file) {
    ofstream myfile;
    myfile.open(output_file);
    for (size_t idx = 0; idx < boundBoxes.size(); idx++) {
        myfile << "Ship 0.8 ";
        myfile << boundBoxes[idx][0];
        myfile << " ";
        myfile << boundBoxes[idx][1];
        myfile << " ";
        myfile << boundBoxes[idx][2];
        myfile << " ";
        myfile << boundBoxes[idx][3];
        myfile << "\n";
    }
    myfile.close();
    return;
}

vector<vector<int> > find_boxes(Mat &image) {
    std::vector<std::vector<cv::Point> > contours;
    Mat contourOutput;
    contourOutput = image.clone();
    
    cv::findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
    std::vector<Rect> boundRect( contours.size() );
    vector<vector<Point> > contours_poly( contours.size() );

    vector<vector<int> > boxes;

    for (size_t idx = 0; idx < contours.size(); idx++) {
        vector<int> box(4);

        cv::approxPolyDP( Mat(contours[idx]), contours_poly[idx], 3, true );
        boundRect[idx] = boundingRect( Mat(contours_poly[idx]) );
        box[0] = boundRect[idx].tl().x;
        box[1] = boundRect[idx].tl().y;
        box[2] = boundRect[idx].br().x - boundRect[idx].tl().x;
        box[3] = boundRect[idx].br().y - boundRect[idx].tl().y;
        if ((box[2] * box[3]) > (MINIMUM_SHIP_SIDE * MINIMUM_SHIP_SIDE)) {
            boxes.push_back(box);
        }
    }

    return boxes;

}

void draw_boxes(vector<vector<int> > &boundBoxes, Mat &bgr_image, cv::Scalar color) {

    for (int idx = 0; idx < boundBoxes.size(); idx++) {
        int l = boundBoxes[idx][0];
        int t = boundBoxes[idx][1];
        int r = boundBoxes[idx][2] + l;
        int b = boundBoxes[idx][3] + t;
        rectangle( bgr_image, Point(l, t), Point(r, b), color, 2, 8, 0 );
    }
}

vector<vector<int> > readGtBoxes(char * gt_file) {
    
    vector<vector<int>> boxes;
    std::ifstream file(gt_file);

    if (file.is_open()) {
    
        std::string line;
        while (getline(file, line)) {
            vector<int> box(4);
            std::stringstream ss(line);
            std::string buf;
            int flag = 0;
            while (ss >> buf) {
                if (flag > 0) {
                    box[flag-1] = std::stoi(buf);
                }
                flag+=1;
            }
            boxes.push_back(box);
        }
        file.close();
    }
    return boxes;
}

void merge_save_image(Mat &im1, Mat &im2, char * output_file) {

    Mat matDst(Size(im1.cols*2 + 150, im1.rows+ + 100),im1.type(),Scalar(255,255,255));

    cv::putText(matDst, "Ground Truth", cv::Point(100, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,255,0), 1, CV_AA);
    cv::putText(matDst, "Detected", cv::Point(im1.cols + 180, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,255), 1, CV_AA);

    Mat matRoi = matDst(Rect(50,50,im2.cols,im2.rows));
    im1.copyTo(matRoi);
    matRoi = matDst(Rect(im1.cols + 100 ,50,im1.cols,im1.rows));
    im2.copyTo(matRoi);
    imwrite( output_file, matDst );
}