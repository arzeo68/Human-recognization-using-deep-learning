//
// Created by alexis on 08/12/2020.
//

#ifndef OPENCV_DEEPLEARNING_HPP
#define OPENCV_DEEPLEARNING_HPP

#include <string>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/core/mat.hpp>

namespace ss
{
    class DeepLearning
    {
        public:
        void displayDetectedRect(cv::Mat &frame);

        DeepLearning(std::string cfgPath, std::string modelPath);

        std::vector<cv::Rect2d> object_detection(cv::Mat &frame);

        std::vector<std::string> getOutputsNames() const;

        std::vector<cv::Rect2d> postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs, bool clear);

        const cv::Mat &getBlob() const;

        void setBlob(const cv::Mat &blob);

        cv::dnn::Net getNet();

        void setNet(const cv::dnn::Net &net);

        float getConfThreshold() const;

        void setConfThreshold(float confThreshold);

        float getNmsThreshold() const;

        void setNmsThreshold(float nmsThreshold);

        int getInpWidth() const;

        void setInpWidth(int inpWidth);

        int getInpHeight() const;

        void setInpHeight(int inpHeight);

        const std::vector<std::string> &getClasses() const;

        void setClasses(const std::vector<std::string> &classes);

        const std::string &getModelPath() const;

        void setModelPath(const std::string &modelPath);

        const std::string &getCfgPath() const;

        void setCfgPath(const std::string &cfgPath);

        private:
        std::vector<cv::Rect2d> _boxesDetected;
        cv::Mat _blob;
        cv::dnn::Net _net;
        float confThreshold = 0.7; // Confidence threshold
        float nmsThreshold = 0.4;  // Non-maximum suppression threshold
        int inpWidth = 416;  // Width of network's input image
        int inpHeight = 416; // Height of network's input image
        std::vector<std::string> classes;
        std::string _modelPath;
        std::string _cfgPath;
    };
}

#endif //OPENCV_DEEPLEARNING_HPP
