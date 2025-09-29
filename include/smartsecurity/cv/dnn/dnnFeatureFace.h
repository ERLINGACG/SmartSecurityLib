//
// Created by HP on 2025/8/17.
//

#ifndef SMART_SECURITY_DNN_FEATURE_FACE_H
#define SMART_SECURITY_DNN_FEATURE_FACE_H
#include "nlohmann/json_fwd.hpp"
#include "smartsecurity/cv/dnn/dnnBasicClass.h"
#include "smartsecurity/cv/param/dnnParam.h"

namespace cv_dnn::face{
    class DnnFeatureFace : public dnnBasic::DnnBasicClass{
        public:
            explicit DnnFeatureFace(param::face::FaceFeatureParam* param);
            explicit DnnFeatureFace(const char* path);
            ~DnnFeatureFace() override =default;
            void InitModelFace(param::face::FaceFeatureParam* param); //
            void InitModelFace(const char* path); //
            void LoadModelFace(param::face::FaceFeatureParam* param) ;

            void LoadJson(const char *path);
            void Load(nlohmann::json& j);

            cv::dnn::Net getNet(){return net;}
    };
}
#endif //SMART_SECURITY_DNN_FEATURE_FACE_H
