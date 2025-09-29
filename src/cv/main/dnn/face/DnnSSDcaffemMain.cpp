//
// Created by HP on 2025/11/14.
//
#include "smartsecurity/cv/cvexport.h"
extern "C"{
    EXPORT_USE CORE_CV_API DnnFaceNet* DnnFaceNetCreate(const char* path){
        return new DnnFaceNet(path);
    }
    EXPORT_USE CORE_CV_API void DnnFaceNetDestroy(const DnnFaceNet* dnnFaceNet){
        delete dnnFaceNet;
    }

    EXPORT_USE CORE_CV_API DnnFaceNet512d* DnnFaceNet512dCreate(const char* path){
        return new DnnFaceNet512d(path);
    }

    EXPORT_USE CORE_CV_API void DnnFaceNet512dDestroy(const DnnFaceNet512d* dnnFaceNet512d){
        delete dnnFaceNet512d;
    }

    EXPORT_USE CORE_CV_API DnnSSDcaffem* DnnSSDcaffemCreate(const char* path,DnnFaceNet* faceNet){
        return new DnnSSDcaffem(path,faceNet);
    }

    EXPORT_USE CORE_CV_API DnnSSDcaffem* DnnSSDcaffemCreateArc(const char* path,DnnArcFace* arcface){
        return new DnnSSDcaffem(path,arcface);
    }

    EXPORT_USE CORE_CV_API DnnSSDcaffem* DnnSSDcaffemCreate512(const char* path,DnnFaceNet512d* faceNet){
        return new DnnSSDcaffem(path,faceNet);
    }


    EXPORT_USE CORE_CV_API void DnnSSDcaffemDestroy(const DnnSSDcaffem* dnnSSDcaffem){
        delete dnnSSDcaffem;
    }
}

extern "C"{
    EXPORT_USE CORE_CV_API void DnnSSDcaffemDebugTime(DnnSSDcaffem* dnnSSDcaffem,int size,unsigned char* img,
        ImageData& OutputData,OutputJson& json
    ){
        dnnSSDcaffem->DebugTime(size,img,OutputData,json);
    }

    EXPORT_USE CORE_CV_API void DetectionFeatureByte(DnnSSDcaffem* dnnSSDcaffem,int size,unsigned char* img,
        ImageData& OutputData,OutputJson& json,FaceFeatureByte& faceFeatureByte
    ){
        dnnSSDcaffem->DetectionFeature(size,img,OutputData,json,faceFeatureByte);
    }


    EXPORT_USE CORE_CV_API void DetectionFeatureByte512d(DnnSSDcaffem* dnnSSDcaffem,int size,unsigned char* img,
        ImageData& OutputData,OutputJson& json,FaceFeatureByte& faceFeatureByte
    ){
        dnnSSDcaffem->DetectionFeature512d(size,img,OutputData,json,faceFeatureByte);
    }


    EXPORT_USE CORE_CV_API void DetectionFeatureByteArc(DnnSSDcaffem* dnnSSDcaffem,int size,unsigned char* img,
        ImageData& OutputData,OutputJson& json,FaceFeatureByte& faceFeatureByte
    ){
        dnnSSDcaffem->DetectionFeatureArc(size,img,OutputData,json,faceFeatureByte);
    }


    EXPORT_USE CORE_CV_API double DnnFaceNetVerifyFeature(DnnFaceNet* dnnFaceNet,unsigned char* input1,unsigned char* input2){
        return dnnFaceNet->VerifyFeature(input1,input2);
    }

    EXPORT_USE CORE_CV_API double DnnFaceNet512dVerifyFeature(DnnFaceNet512d* dnnFaceNet512d,unsigned char* input1,unsigned char* input2){
        return dnnFaceNet512d->VerifyFeature512d(input1,input2);
    }

}