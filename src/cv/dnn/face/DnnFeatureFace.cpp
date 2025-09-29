//
// Created by HP on 2025/8/20.
//
#include <iostream>
#include <fstream>
#include "smartsecurity/cv/dnn/dnnFeatureFace.h"
#include "nlohmann/json.hpp"
using  namespace cv_dnn::face;

DnnFeatureFace::DnnFeatureFace(param::face::FaceFeatureParam* param) {
    this->InitModelFace(param);
}
void DnnFeatureFace::InitModelFace(param::face::FaceFeatureParam* param) {
    this->LoadModelFace(param);
}

void DnnFeatureFace::LoadModelFace(param::face::FaceFeatureParam* param) {
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
    nlohmann::json activity=j["env"];
    json env=j[activity];
    std::cout<<"env:"<<env<<std::endl;
    std::cout<<"modelPath:"<<env["facenet_path"]<<std::endl;
    std::cout<<"isCuda:"<<env["isCuda"]<<std::endl;
    try{
        this->net =cv::dnn::readNetFromONNX(std::string(env["facenet_path"]));
        if(env["isCuda"]) {
            this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout<<"dnn backend cuda facenet"<<std::endl;
        }
        std::cout<<"facenet laod success"<<std::endl;
    }catch (const std::exception& e){
        std::cout<<"load model failed"<<e.what()<<std::endl;
    }
}

void DnnFeatureFace::InitModelFace(const char *path) {
    this->LoadJson(path);
}
