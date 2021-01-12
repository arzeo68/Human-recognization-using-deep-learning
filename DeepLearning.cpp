//
// Created by alexis on 08/12/2020.
//

#include "DeepLearning.hpp"

namespace ss
{

    void DeepLearning::displayDetectedRect(cv::Mat &frame)
    {
        for (auto &obj : _boxesDetected)
        {
            rectangle(frame, obj, cv::Scalar(0, 255, 0), 2, 1);
        }
    }

    std::vector<cv::Rect2d> DeepLearning::object_detection(cv::Mat &frame)
    {
        if (frame.empty())
        {
            return {};
        }
        cv::dnn::blobFromImage(frame, _blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);
        _net.setInput(_blob);

        std::vector<cv::Mat> outs;
        _net.forward(outs, getOutputsNames());
        // postprocess to get object boxes and add it to tracker
        _boxesDetected = postprocess(frame, outs, true);
        return _boxesDetected;
    }

    std::vector<std::string> DeepLearning::getOutputsNames() const
    {
        static std::vector<std::string> names;
        if (names.empty())
        {
            //Get the indices of the output layers, i.e. the layers with unconnected outputs
            std::vector<int> outLayers = _net.getUnconnectedOutLayers();

            //get the names of all the layers in the network
            std::vector<std::string> layersNames = _net.getLayerNames();

            // Get the names of the output layers in names
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i)
                names[i] = layersNames[outLayers[i] - 1];
        }
        return names;
    }

    std::vector<cv::Rect2d> DeepLearning::postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs, bool clear)
    {
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect2d> boxes;

        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            float *data = (float *)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.emplace_back(cv::Rect(left, top, width, height));
                }
            }
        }

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        if (clear)
        {
            std::vector<cv::Rect2d> rectCleaned;
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
            for (size_t i = 0; i < indices.size(); ++i)
            {
                int idx = indices[i];

                if (classes[classIds[idx]] == "person" || classes[classIds[idx]] == "sports ball")
                {
                    putText(frame, classes[classIds[idx]], cv::Point(boxes[idx].x, boxes[idx].y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(1,0,0),2);
                    rectCleaned.push_back(boxes[idx]);
                }

            }
            return rectCleaned;
        }
        return boxes;
    }

    const cv::Mat &DeepLearning::getBlob() const
    {
        return _blob;
    }

    void DeepLearning::setBlob(const cv::Mat &blob)
    {
        _blob = blob;
    }

    cv::dnn::Net DeepLearning::getNet()
    {
        return _net;
    }

    void DeepLearning::setNet(const cv::dnn::Net &net)
    {
        _net = net;
    }

    float DeepLearning::getConfThreshold() const
    {
        return confThreshold;
    }

    void DeepLearning::setConfThreshold(float confThreshold)
    {
        DeepLearning::confThreshold = confThreshold;
    }

    float DeepLearning::getNmsThreshold() const
    {
        return nmsThreshold;
    }

    void DeepLearning::setNmsThreshold(float nmsThreshold)
    {
        DeepLearning::nmsThreshold = nmsThreshold;
    }

    int DeepLearning::getInpWidth() const
    {
        return inpWidth;
    }

    void DeepLearning::setInpWidth(int inpWidth)
    {
        DeepLearning::inpWidth = inpWidth;
    }

    int DeepLearning::getInpHeight() const
    {
        return inpHeight;
    }

    void DeepLearning::setInpHeight(int inpHeight)
    {
        DeepLearning::inpHeight = inpHeight;
    }

    const std::vector<std::string> &DeepLearning::getClasses() const
    {
        return classes;
    }

    void DeepLearning::setClasses(const std::vector<std::string> &classes)
    {
        DeepLearning::classes = classes;
    }

    const std::string &DeepLearning::getModelPath() const
    {
        return _modelPath;
    }

    void DeepLearning::setModelPath(const std::string &modelPath)
    {
        DeepLearning::_modelPath = modelPath;
    }

    const std::string &DeepLearning::getCfgPath() const
    {
        return _cfgPath;
    }

    void DeepLearning::setCfgPath(const std::string &cfgPath)
    {
        DeepLearning::_cfgPath = cfgPath;
    }

    DeepLearning::DeepLearning(std::string cfgPath, std::string modelPath): _cfgPath(cfgPath), _modelPath(modelPath)
    {

        // Load the network
        _net = cv::dnn::readNetFromDarknet(cfgPath, modelPath);
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
}
