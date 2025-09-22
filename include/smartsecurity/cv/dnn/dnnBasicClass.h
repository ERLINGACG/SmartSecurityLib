//
// Created by HP on 2025/8/15.
//

#ifndef SMARTSECURITYCORELIB_DNN_BASIC_CLASS_H
#define SMARTSECURITYCORELIB_DNN_BASIC_CLASS_H
#include <opencv2/dnn.hpp>
#include "opencv2/mcc/checker_model.hpp"
#include "smartsecurity/cv/dnn/dnnInterface.h"


namespace cv_dnn::dnnBasic{
class DnnBasicClass:public cv_dnn::interface::DnnInterface{ //隔离接口，均为空实现，软约定
    public:
       cv::dnn::Net net;
    public:
        ~DnnBasicClass() override =default;

        void LoadJson(const char *path) override {};
        void Load(nlohmann::json& j) override {};
        void inputImage(unsigned char *inputData, int size, cv::Mat &orgImage) override{}
        void DetectImage(unsigned char* inputData,int size,data::ImageData& OutputData) override {};
        void SetBlob(cv::Mat& blob,cv::Mat& inputImg) override {};
        void Forward(cv::Mat& output) override {};
        void Forward(std::vector<cv::Mat>& output_mat) override {};
        void ProcessResults(cv::Mat& inputImg,
                             std::vector<cv::Mat>& output_mat,
                             std::vector<int>& class_ids,//分类类别索引
                             std::vector<float>& confidences,//置信度
                             std::vector<cv::Rect>& boxes,//边框坐标信息
                             std::vector<int>& nms_result) override {};
        void ProcessResults(
                cv::Mat &output,
                cv::Mat& orgImage,
                double confidence_threshold,
                std::vector<cv::Rect>& _faces
                ) override{};

    };
}
#endif //SMARTSECURITYCORELIB_DNN_BASIC_CLASS_H
