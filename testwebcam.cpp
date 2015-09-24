#include <iostream>

#include "webcamcontroller.hpp"

static Mat frame;
static const int FPS = 10;

static string getMouseLocation (int x, int y)
{
    stringstream ss;
    ss << "x = " << x << ", y = " << y;
    return ss.str ();
}

static void cropDisplay (int x, int y, int width, int height)
{
    Rect roi = Rect (x, y, width, height);
    Mat cropped = frame (roi);

    namedWindow ("crop", CV_WINDOW_AUTOSIZE);
    imshow ("crop", cropped);
}

static void onMouse (int event, int x, int y, int flags, void *userdata)
{
    string location = getMouseLocation (x, y);
    
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
                cropDisplay (left, top, right - left, bottom - top);
            }
            selecting = !selecting;
        }
    }
}

int main (void)
{
    WebcamController *controller = new (WebcamController);

    namedWindow ("output", CV_GUI_EXPANDED);
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
