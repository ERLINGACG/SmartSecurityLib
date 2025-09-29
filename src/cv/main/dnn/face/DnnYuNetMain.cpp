//
// Created by HP on 2025/11/20.
//
#include "smartsecurity/cv/cvexport.h"
using cv_dnn::DnnYuNet;
extern "C"{
   EXPORT_USE CORE_CV_API DnnYuNet* DnnYuNetCreate(const char* configPath){
      return new DnnYuNet(configPath);
   }
   EXPORT_USE CORE_CV_API void DnnYuNetDestroy(const DnnYuNet* net){
      delete net;
   }

}

extern "C"{
      EXPORT_USE CORE_CV_API void DnnYuNetGetInfo(DnnYuNet* net){
         net->getInfo();
      }
      EXPORT_USE CORE_CV_API void DnnYuNetSetArc(DnnYuNet* net,DnnArcFace* face)
      {
         net->setArcFace(face);
      }
      EXPORT_USE CORE_CV_API void DnnYuNetDebugTime(DnnYuNet* net,int size, unsigned char* img, ImageData& OutputData, OutputJson& json,FaceFeatureByte& faceFeatureByte){
         net->DebugTime(size,img,OutputData,json,faceFeatureByte);
      }
}