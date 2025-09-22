//
// Created by HP on 2025/8/15.
//
#include "smartsecurity/cv/cvexport.h"
using  namespace cv_dnn::face;

extern "C"{
    EXPORT_USE CORE_CV_API DnnDetectorFace* DnnDetectorFaceCreate(cv_dnn::param::face::FaceParam* param){
        return new DnnDetectorFace(param);
    }
    EXPORT_USE CORE_CV_API DnnFeatureFace* DnnFeatureFaceCreate(cv_dnn::param::face::FaceFeatureParam* param){
        return new DnnFeatureFace(param);
    }
    EXPORT_USE CORE_CV_API DnnDetectorFace* DnnDetectorFaceCreate_1(const char* path){
        return new DnnDetectorFace(path);
    }
}
extern "C"{
    EXPORT_USE CORE_CV_API void DnnDetectorFaceGetFaceFeature_0(
            DnnDetectorFace* dnnDetectorFace,
            unsigned char* inputData,
            int size,
            cv_param::EncodeParam* encodeParam,
            cv_dnn::face::DnnFeatureFace* faceFeature,
            data::ImageData& OutputData,
            data::FaceFeatureByte  & faceFeatureByte
    ){
        dnnDetectorFace->getFaceFeature(
                inputData,size,
                encodeParam,
                faceFeature,
                OutputData,faceFeatureByte);
    }
    EXPORT_USE CORE_CV_API double DnnDetectorFaceGetDistanceForByte(unsigned char* input_1, unsigned char* input_2){
        return DnnDetectorFace::getDistanceForByte(input_1,input_2);
    }
}
