//
// Created by HP on 2025/8/20.
//
#include <iostream>
#include "smartsecurity/cv/dnn/dnnFeatureFace.h"

using  namespace cv_dnn::face;

DnnFeatureFace::DnnFeatureFace(cv_dnn::param::face::FaceFeatureParam* param) {
    this->InitModelFace(param);
}
void DnnFeatureFace::InitModelFace(cv_dnn::param::face::FaceFeatureParam* param) {
    this->LoadModelFace(param);
}

void DnnFeatureFace::LoadModelFace(cv_dnn::param::face::FaceFeatureParam* param) {
    this->net = cv::dnn::readNetFromONNX(param->modelPath);
    if(param->isCuda){
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout<<"dnn backend cuda"<<std::endl;
    }
}
