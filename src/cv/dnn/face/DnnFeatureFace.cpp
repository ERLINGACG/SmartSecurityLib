//
// Created by HP on 2025/8/20.
//
#include <iostream>
#include <fstream>
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

DnnFeatureFace::DnnFeatureFace(const char *path) {
    this->InitModelFace(path);
}

void DnnFeatureFace::LoadJson(const char *path) {
    std::unique_ptr<std::ifstream> jsonfile(new std::ifstream(path));
    if(jsonfile->is_open()){
        nlohmann::json j;
        *jsonfile >> j;
        try{
            Load(j);
        }catch(const std::exception& e){
            std::cout<<"load config file failed"<<e.what()<<std::endl;
        }

    }else{
        std::cout<<"no config file "<<path<<std::endl;
    }
}

void DnnFeatureFace::Load(nlohmann::json &j) {

}

void DnnFeatureFace::InitModelFace(const char *path) {
    this->LoadJson(path);
}
