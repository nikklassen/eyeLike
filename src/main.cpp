#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <numeric>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

#define SAMPLE_SIZE 5

/** Constants **/


/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

cv::Point leftPupil, rightPupil;
std::vector<std::pair<cv::Point, cv::Point> > bounds;
cv::Point gazePoint;

const cv::Point P1(640, 400), P2(1200, 700);

double H_SCREEN_WIDTH = 640.0, H_SCREEN_HEIGHT = 400.0;

cv::Mat *colorFrame;

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
  if (event == cv::EVENT_LBUTTONDOWN) {
    if (bounds.size() != 1 || (leftPupil.x != bounds.at(0).first.x && leftPupil.y != bounds.at(0).first.y)) {
      std::cout << "Saving point " << leftPupil << std::endl;
      bounds.push_back(std::make_pair(leftPupil, rightPupil));
    }
  }
}

/**
 * @function main
 */
int main( int argc, const char** argv ) {
  CvCapture* capture;
  cv::Mat frame;

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };

  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 0, 0);
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(face_window_name, 10, 100);
  //cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
  //cv::moveWindow("Right Eye", 10, 600);
  //cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
  //cv::moveWindow("Left Eye", 10, 800);
  //cv::namedWindow("aa",CV_WINDOW_NORMAL);
  //cv::moveWindow("aa", 10, 800);
  //cv::namedWindow("aaa",CV_WINDOW_NORMAL);
  //cv::moveWindow("aaa", 10, 800);

  cv::setMouseCallback(main_window_name, mouseCallback, NULL);

  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155), cv::Size(23, 15),
      43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

  // Read the video stream
  capture = cvCaptureFromCAM( -1 );
  cv::Point savedGazePoint(0, 0);
  if( capture ) {
    while( true ) {
      frame = cvQueryFrame( capture );
      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(debugImage);
      colorFrame= &debugImage;

      // Apply the classifier to the frame
      if( !frame.empty() ) {
        detectAndDisplay( frame );
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      std::stringstream ss;
      ss << savedGazePoint;

      putText(debugImage, ss.str(), cv::Point(640, 400), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
      if (bounds.size() == 0) {
        circle(debugImage, P1, 3, 1234);
      } else if (bounds.size() == 1) {
        circle(debugImage, P2, 3, 1234);
      }

      imshow(main_window_name,debugImage);

      char c = (char) cv::waitKey(10);
      if( c == 'c' ) { break; }
      else if( c == 'f' ) {
        imwrite("frame.png",frame);
      } else if (c == 'p') {
        savedGazePoint = gazePoint;
      } else if (c == 'r') {
        bounds.clear();
      }

    }
  }

  releaseCornerKernels();

  return 0;
}

template<class T> T boundVal(T min, T max, T val) {
  return std::max(std::min(max, val), min);
}

cv::Point getGazePoint(cv::Point c, cv::Point br, cv::Point pupil) {

  const double MAX_SCREEN_R = sqrt(pow(H_SCREEN_WIDTH, 2) + pow(H_SCREEN_HEIGHT, 2));

  double dx = br.x - c.x,
         dy = br.y - c.y;

  double cornerR = sqrt(pow(dx, 2) + pow(dy, 2));

  dx = pupil.x - c.x,
     dy = pupil.y - c.y;
  double theta = atan2(dy, dx);
  double r = sqrt(pow(dx, 2) + pow(dy, 2));

  double screenR = r / cornerR * MAX_SCREEN_R,
         screenX = boundVal(-1 * H_SCREEN_WIDTH, H_SCREEN_WIDTH, screenR * cos(theta)) + H_SCREEN_WIDTH,
         screenY = boundVal(-1 * H_SCREEN_HEIGHT, H_SCREEN_HEIGHT, screenR * sin(theta)) + H_SCREEN_HEIGHT;

  return cv::Point(screenX, screenY);
}

cv::Point cornerVec(cv::Point corner, cv::Point pupil) {
  // Positive y is down, dx is always positive
  return cv::Point(std::abs(corner.x - pupil.x), pupil.y - corner.y);
}

cv::Point getGazePointWithCorners(cv::Point corner, cv::Point pupil) {
  // Assume left eye
  // gaze angle is coordinates
  double a1 = P1.x, //bounds.at(0).first.x,
         a2 = P2.x,
         b1 = P1.y - H_SCREEN_HEIGHT, //bounds.at(1).first.y;
         b2 = P2.y - H_SCREEN_HEIGHT;
  cv::Point v1 = cornerVec(corner, bounds.at(0).first),
    v2 = cornerVec(corner, bounds.at(1).first),
    v = cornerVec(corner, pupil);

  double x = boundVal(0.0, 2 * H_SCREEN_WIDTH, a1 + (v.x - v1.x) / (v2.x - v1.x) * (a2 - a1)),
         y = b1 + (v.y - v1.y) / (v2.y - v1.y) * (b2 - b1);

  return cv::Point(x, y + H_SCREEN_HEIGHT);
}

cv::Point addPoints(cv::Point a, cv::Point b) {
  return cv::Point(a.x + b.x, a.y + b.y);
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
  cv::Mat faceROI = frame_gray(face);
  cv::Mat debugFace = faceROI;
  static std::deque<cv::Point> gazePoints;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
      eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
      eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  // get corner regions
  cv::Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  cv::Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  cv::Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  cv::Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);

  cv::Point newGazePoint(0, 0);

  //-- Find Eye Corners
  if (kEnableEyeCorner) {
    cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
    cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
    cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
    cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
    circle(faceROI, leftRightCorner, 3, 200);
    circle(faceROI, leftLeftCorner, 3, 200);
    circle(faceROI, rightLeftCorner, 3, 200);
    circle(faceROI, rightRightCorner, 3, 200);

    if (bounds.size() >= 2) {
      cv::Point leftLeftGaze = getGazePointWithCorners(leftLeftCorner, leftPupil),
                leftRightGaze = getGazePointWithCorners(leftRightCorner, leftPupil),
                avgGaze = addPoints(leftLeftGaze, leftRightGaze);

      newGazePoint = cv::Point(avgGaze.x / 2.0, avgGaze.y / 2.0);
    }
  } else if (bounds.size() >= 2) {
    cv::Point leftC = bounds.at(0).first;
    cv::Point leftBr = bounds.at(1).first;
    cv::Point leftGaze = getGazePoint(leftC, leftBr, leftPupil);

    cv::Point rightC = bounds.at(0).second;
    cv::Point rightBr = bounds.at(1).second;
    cv::Point rightGaze = getGazePoint(rightC, rightBr, rightPupil);

    double screenX = (leftGaze.x + rightGaze.x) / 2.0,
           screenY = (leftGaze.y + rightGaze.y) / 2.0;

    newGazePoint = cv::Point(screenX, screenY);
  }

  gazePoints.push_back(newGazePoint);
  if (gazePoints.size() > SAMPLE_SIZE) {
    gazePoints.pop_front();
  }

  cv::Point gazePointsTotal = std::accumulate(gazePoints.begin(), gazePoints.end(), cv::Point(0, 0), addPoints);
  gazePoint = cv::Point(gazePointsTotal.x / gazePoints.size(), gazePointsTotal.y / gazePoints.size());

  circle(*colorFrame, gazePoint, 3, 200);

  imshow(face_window_name, faceROI);
  //  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
  //  cv::Mat destinationROI = debugImage( roi );
  //  faceROI.copyTo( destinationROI );
}


cv::Mat findSkin (cv::Mat &frame) {
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
//    uchar *Or = output.ptr<uchar>(y);
    cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      cv::Vec3b ycrcb = Mr[x];
//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = cv::Vec3b(0,0,0);
      }
    }
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) {
  std::vector<cv::Rect> faces;
  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];

  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
//  findSkin(debugImage);

  for( int i = 0; i < faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], 1234);
  }
  //-- Show what you got
  if (faces.size() > 0) {
    findEyes(frame_gray, faces[0]);
  }
}
