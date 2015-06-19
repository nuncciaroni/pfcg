#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;

float Distance(Point p1, Point p2)
{
    return hypot(p1.x - p2.x, p1.y - p2.y);
}

bool Collision(Point p1, float rayP1, Point p2, float rayP2)
{
    if(Distance(p1, p2) < (rayP1 + rayP2))
        return true;
    return false;
}

int main()
{
    VideoCapture cap(1); //capture the video from web cam

    //Point winSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    Point winSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), 0);
    float raio1 = 80;
    float raio2 = 150;

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
        cap.read(imgOriginal); // read a new frame from video

        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

        detector.detect(imgThresholded, keypoints);

        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));

        Moments objMoments = moments(imgThresholded);
        float dM01 = objMoments.m01;
        float dM10 = objMoments.m10;
        float dArea = objMoments.m00;
        if (dArea > 10000)
        {
            cout << "X " << dM10/dArea << endl;
            cout << "Y " << dM01/dArea << endl;
            if (Collision(Point(dM10/dArea, dM01/dArea), raio1, winSize, raio2))
                cout << "Objeto dentro de area " << endl;
        }

        drawKeypoints(imgOriginal, keypoints, img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

        circle(imgOriginal, Point(dM10/dArea, dM01/dArea), raio1, Scalar(0, 0, 255), 3);
        circle(imgOriginal,winSize, raio2, Scalar(0, 0, 255), 3);
        imshow("Thresholded Image", imgThresholded); //show the thresholded image
        imshow("Original", imgOriginal); //show the original image

        cout << (keypoints.size() > 5 ? "objeto detectado" : " ") << endl;

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }

    return 0;

}
