#ifndef TEST_WEBCAM_H_
#define TEST_WEBCAM_H_

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class WebcamController
{
public:
  WebcamController ();

  bool getPaused (void) const;
  void setPaused (bool paused);
  Mat getFrame (void);
  Mat getEdges (void);

private:
  VideoCapture cap;
  Mat savedFrame;
  bool paused;
};

#endif // TEST_WEBCAM_H_
