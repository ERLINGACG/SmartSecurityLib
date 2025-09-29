//
// Created by HP on 2025/11/19.
//
#include "smartsecurity/cv/cvexport.h"


extern "C" {

    CORE_CV_API DnnArcFace* DnnArcFaceCreate(const char* path){
        return new DnnArcFace(path);
    }
    CORE_CV_API void DnnArcFaceDestroy(DnnArcFace* dnnArcFace){
        delete dnnArcFace;
    }
    CORE_CV_API double DnnArcFaceVerifyFeature(DnnArcFace* dnnArcFace,unsigned char* input1 ,unsigned char* input2){
        return   dnnArcFace->ArcVerifyFeature512(input1,input2);
    }
}