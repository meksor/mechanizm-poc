#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
int main()
{
  VideoCapture capture(samples::findFile("test.mp4"));
  if (!capture.isOpened())
  {
    //error in opening the video input
    cerr << "Unable to open file!" << endl;
    return 0;
  }
  int width = capture.get(CAP_PROP_FRAME_WIDTH);
  int height = capture.get(CAP_PROP_FRAME_HEIGHT);
  std::string image_path = samples::findFile("test.png");
  Mat prevNewFrame = imread(image_path, IMREAD_COLOR);
  resize(prevNewFrame, prevNewFrame, cv::Size(width, height), 0, 0, INTER_LINEAR);

  Mat frame1, prvs, newFrame;
  capture >> frame1;
  cvtColor(frame1, prvs, COLOR_BGR2GRAY);
  while (true)
  {
    Mat frame2, next;
    capture >> frame2;

    if (frame2.empty())
      break;

    cvtColor(frame2, next, COLOR_BGR2GRAY);
    Mat flow(prvs.size(), CV_32FC2);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    Mat map(flow.size(), CV_32FC2);
    for (int y = 0; y < map.rows; ++y)
    {
      for (int x = 0; x < map.cols; ++x)
      {
        Point2f f = flow.at<Point2f>(y, x);
        map.at<Point2f>(y, x) = Point2f(x + f.x, y + f.y);
      }
    }
    remap(prevNewFrame, newFrame, map, Mat(), INTER_LINEAR);
    imshow("frame2", newFrame);
    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27)
      break;
    prvs = next;
    prevNewFrame = newFrame;
  }
}