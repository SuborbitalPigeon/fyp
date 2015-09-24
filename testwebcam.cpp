#include "webcamcontroller.hpp"

static const int FPS = 10;

int main (void)
{
    WebcamController *controller = new (WebcamController);

    namedWindow ("output", CV_GUI_EXPANDED);

    while (true)
    {
        Mat frame;

        frame = controller->getFrame ();

        imshow ("output", frame);
        int c = waitKey (1000 / FPS);
        switch ((char) c)
        {
            case 'p':
                bool isPaused = controller->getPaused ();
                controller->setPaused (!isPaused);
                break;
        }
    }

    return 0;
}
