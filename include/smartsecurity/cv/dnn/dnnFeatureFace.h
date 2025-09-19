//
// Created by HP on 2025/8/17.
//

#ifndef SMARTSECURITYCORELIB_DNN_FEATURE_FACE_H
#define SMARTSECURITYCORELIB_DNN_FEATURE_FACE_H
#include "smartsecurity/cv/dnn/dnnBasicClass.h"
#include "smartsecurity/cv/param/dnnParam.h"
namespace cv_dnn::face{
    class DnnFeatureFace : public cv_dnn::dnnBasic::DnnBasicClass{
        public:
            explicit DnnFeatureFace(cv_dnn::param::face::FaceFeatureParam* param);
            ~DnnFeatureFace() override=default;;
            void InitModelFace(cv_dnn::param::face::FaceFeatureParam* param);
            void LoadModelFace(cv_dnn::param::face::FaceFeatureParam* param) ;
            void LoadJson(const char *path) override{};
            void SetBlob(cv::Mat& blob,cv::Mat& inputImg) override{};
            void DetectImage(unsigned char* inputData, int size, data::ImageData& OutputData) override{};
            cv::dnn::Net getNet(){return net;}
    };
}
#endif //SMARTSECURITYCORELIB_DNN_FEATURE_FACE_H
