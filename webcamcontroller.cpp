#include "webcamcontroller.hpp"

using namespace cv;
using namespace std;

static string getDateTime (void)
{
    time_t rawtime;
    tm *timeinfo;
    char buffer[20];

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime (buffer, sizeof(buffer), "%F-%T", timeinfo);

    string ret = buffer;
    return ret;
}

WebcamController::WebcamController()
{
    this->cap = VideoCapture (0);
    this->paused = false;
}

void WebcamController::saveImage (void)
{
    Mat image = this->getFrame ();
    Mat edges = this->getEdges ();

    // Save as a PNG with speed parameter of 9
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    string dateTime = getDateTime ();

    // standard
    string filename = "output-";
    filename.append (dateTime);
    filename.append (".png");
    imwrite (filename, image);

    // edges
    string edgename = "edges-";
    edgename.append (dateTime);
    edgename.append (".png");
    imwrite (edgename, edges);
;
}

bool WebcamController::getPaused (void) const
{
    return this->paused;
}

void WebcamController::setPaused (bool paused)
{
    Mat frame;
    Mat edges;

    if (paused == true)
        frame = this->getFrame ();

    this->paused = paused;
}

Mat WebcamController::getFrame (void)
{
    Mat ret;

    if (this->paused == true)
        return this->savedFrame;
    else
        this->cap.read (ret);
        this->savedFrame = ret;
        return ret;
}

Mat WebcamController::getEdges (void)
{
    Mat frame = this->getFrame ();
    Mat edges;

    cvtColor (frame, edges, COLOR_BGR2GRAY);
    Canny (edges, edges, 10, 30, 3, 5);
    return edges;
}
