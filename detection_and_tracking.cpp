
// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <iostream>
#include <chrono>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include "DeepLearning.hpp"

const char *keys =
        "{help h usage ? | | Usage examples: ./object_detection_yolo.out --video=run_sm.mp4}"
        "{image i        |<none>| input image   }"
        "{video v       |<none>| input video   }"
;
using namespace cv;
using namespace dnn;
using namespace std::chrono;
using namespace std;

// Initialize the parameters
int main(int argc, char** argv)
{
    ss::DeepLearning deepLearning("../yolov3.cfg", "../yolov3.weights");

    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    // Load names of classes
    string classesFile = "../coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    std::vector<std::string> tmp;
    while (getline(ifs, line))
        tmp.push_back(line);
    deepLearning.setClasses(tmp);

    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame;
    Mat detectedFrame;

    try {
        outputFile = "yolo_out_cpp.avi";
        if (parser.has("video")) {
            // Open the video file
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile)
                throw("error");
            cap.open(str);
            str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;
        } else cap.open(parser.get<int>("device"));
    } catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }

    // Get the video writer initialized to save the output video
    if (!parser.has("image")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }

    // Create a window
    static const string kWinName = "Object detection + tracking";
    namedWindow(kWinName, WINDOW_NORMAL);

    // First Detection
    cap >> frame;
    if (frame.empty()) {
        cap.release();
        return 84;
    }
    imshow(kWinName, frame);

    bool detect = false;
    // Process frames.
    while (true) {
        int64_t timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        detect = !detect;
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }

        deepLearning.object_detection(frame);
        deepLearning.displayDetectedRect(frame);
        // Write the frame with the detection boxes to output video or image
//        frame.convertTo(detectedFrame, CV_8U);
//        if (parser.has("image"))
//            imwrite(outputFile, detectedFrame);
//        else
//            video.write(detectedFrame);
        int64_t t = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - timestamp;
        string label = format("Inference time for a frame : %.2f ms", t);
        std::cout << t << std::endl;
        imshow(kWinName, frame);

        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    if (!parser.has("image"))
        video.release();
    return 0;
}