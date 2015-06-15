#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <math.h>

using namespace cv;
using namespace std;

float Distance(Point2f p1, Point2f p2)
{
    float d = 0;
    d = sqrt(((p2.x - p1.x)*(p2.x - p1.x)) + ((p2.y - p1.y)*(p2.y - p1.y)));
    return d;
}

int main( int argc, char** argv )
{
    VideoCapture cap(1); //capture the video from web cam

    int minHessian = 400;
    Mat img;
    vector<KeyPoint> keypoints, aglomerate;

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

        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5)));
        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5)));

        detector.detect(imgThresholded, keypoints);

        int tem = keypoints.size();

    for(int i = 0; i < tem; i++)
    {
      for(int j = tem-1; j == i; j--)
      {
        if(Distance(keypoints[i].pt, keypoints[j].pt) < 300)
        {
          aglomerate.push_back(keypoints[j]);
        }
      }
    }

            drawKeypoints(imgOriginal, aglomerate, img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

            imshow("Thresholded Image", imgThresholded); //show the thresholded image
            imshow("Original", img); //show the original image

            cout << (aglomerate.size() > 5 ? "objeto detectado" : " ") << endl;

            if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
            {
                cout << "esc key is pressed by user" << endl;
                break;
            }
        }
        keypoints.clear();

        return 0;

    }
