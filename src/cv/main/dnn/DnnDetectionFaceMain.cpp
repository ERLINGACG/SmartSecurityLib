//
// Created by HP on 2025/8/15.
//
#include "smartsecurity/cv/cvexport.h"
using  namespace cv_dnn::face;

extern "C"{

    EXPORT_USE CORE_CV_API DnnFeatureFace* DnnFeatureFaceCreate(const char* path){
        return new DnnFeatureFace(path);
    }
    EXPORT_USE CORE_CV_API DnnDetectorFace* DnnDetectorFaceCreate(const char* path){
        return new DnnDetectorFace(path);
    }

    EXPORT_USE CORE_CV_API void DnnDetectorFaceDestroy(DnnDetectorFace* dnnDetectorFace){
        delete dnnDetectorFace;
    }
    EXPORT_USE CORE_CV_API void DnnFeatureFaceDestroy(DnnFeatureFace* dnnFeatureFace){
        delete dnnFeatureFace;
    }

}
extern "C"{
    EXPORT_USE CORE_CV_API void DnnDetectorFaceGetFaceFeature(
            DnnDetectorFace* dnnDetectorFace,
            unsigned char* inputData,
            int size,
            cv_param::EncodeParam* encodeParam,
            DnnFeatureFace* faceFeature,
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
