#include <iostream>

#include "webcamcontroller.hpp"

static const int FPS = 10;
static WebcamController *controller;

static string locationStr (int x, int y)
{
    stringstream ss;
    ss << "x = " << x << ", y = " << y;
    return ss.str ();
}

static void onMouse (int event, int x, int y, int flags, void *userdata)
{
    string location = locationStr (x, y);
    
    // For cropping operations
    static bool selecting = false;
    static int top, bottom, left, right;
    
    switch (event)
    {
        case EVENT_MOUSEMOVE:
        {
            displayStatusBar ("output", location, 0);
            break;
        }
        case EVENT_LBUTTONDOWN:
        {
            // This is a new selection
            if (selecting == false)
            {
                left = x;
                top = y;
                displayStatusBar ("output", "DEBUG: Button pressed at " + location, 1000);
            }
            else
            {
                right = x;
                bottom = y;
                displayStatusBar ("output", "DEBUG: Button pressed at " + location, 1000);
                // TODO manage starting crop from other corners
                Mat crop = controller->getCropped (left, top, right - left, bottom - top);
                int centrex = (right - left) / 2;
                int centrey = (bottom - top) / 2;

                namedWindow ("crop", CV_WINDOW_AUTOSIZE);
                displayStatusBar ("crop", "Centre = " + locationStr (centrex, centrey), 0);
                imshow ("crop", crop);

            }
            selecting = !selecting;
        }
    }
}

int main (void)
{
    Mat frame;
    controller = new (WebcamController);

    namedWindow ("output", CV_WINDOW_AUTOSIZE | CV_GUI_EXPANDED);
    setMouseCallback ("output", onMouse, 0);

    while (true)
    {
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
