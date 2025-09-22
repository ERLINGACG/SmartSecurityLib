//
// Created by HP on 2025/8/15.
//

#ifndef SMARTSECURITYCORELIB_DNN_INTERFACE_H
#define SMARTSECURITYCORELIB_DNN_INTERFACE_H
#include "smartsecurity/data/image_data.h"
#include "nlohmann/json.hpp"


namespace cv_dnn::interface{
    class DnnInterface{
    public:

        virtual ~DnnInterface()=default;
//        virtual void Load(nlohmann::json& j)=0;
        virtual void LoadJson(const char* path)=0;
        virtual void Load(nlohmann::json& j)=0;
        virtual void SetBlob(cv::Mat& blob,cv::Mat& inputImg)=0;
        virtual void Forward(cv::Mat& output)=0;
        virtual void Forward(std::vector<cv::Mat>& output_mat)=0;
        virtual void inputImage(unsigned char *inputData, int size, cv::Mat &orgImage)=0;
        virtual void ProcessResults( cv::Mat& inputImg,
                                     std::vector<cv::Mat>& output_mat,
                                     std::vector<int>& class_ids,//分类类别索引
                                     std::vector<float>& confidences,//置信度
                                     std::vector<cv::Rect>& boxes,//边框坐标信息
                                     std::vector<int>& nms_result)=0;

        virtual void ProcessResults( cv::Mat &output,
                                     cv::Mat& orgImage,
                                     double confidence_threshold,
                                     std::vector<cv::Rect>& _faces)=0;
        virtual void DetectImage(unsigned char* inputData,int size,data::ImageData& OutputData)=0;

    };
}
#endif //SMARTSECURITYCORELIB_DNN_INTERFACE_H
