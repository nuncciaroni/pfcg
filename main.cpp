#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    VideoCapture cap(1); //capture the video from web cam

    int minHessian = 400;
    Mat img;
    vector<KeyPoint> keypoints;

    SurfFeatureDetector detector(minHessian);
    Mat imgOriginal;
    Mat imgHSV;
    Mat imgThresholded;

    cvNamedWindow("Thresholded Image");
    cvNamedWindow("Original");

    int iLowH = 166;
    int iHighH = 178;

    int iLowS = 101;
    int iHighS = 255;

    int iLowV = 40;
    int iHighV = 255;

    while (true)
    {
        bool bSuccess = cap.read(imgOriginal); // read a new frame from video

        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

        detector.detect(imgThresholded, keypoints);

        drawKeypoints(imgOriginal, keypoints, img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

        imshow("Thresholded Image", imgThresholded); //show the thresholded image
        imshow("Original", img); //show the original image

        cout << (keypoints.size() > 5 ? "objeto detectado" : " ") << endl;

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }

    return 0;

}
