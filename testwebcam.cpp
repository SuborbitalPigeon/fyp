#include "webcamcontroller.hpp"

static const int FPS = 10;

int main (void)
{
    WebcamController *controller = new (WebcamController);

    namedWindow ("output", CV_GUI_EXPANDED);
    namedWindow ("edges", CV_GUI_EXPANDED);

    while (true)
    {
        Mat frame;
        Mat edges;

        frame = controller->getFrame ();
        edges = controller->getEdges ();

        imshow ("output", frame);
        imshow ("edges", edges);

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
