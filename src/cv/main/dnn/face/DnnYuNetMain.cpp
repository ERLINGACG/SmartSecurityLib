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

     EXPORT_USE CORE_CV_API double DnnYuNetVerifyFeature(unsigned char* input1, unsigned char* input2){
         auto* feat1 = reinterpret_cast<float*>(input1);
         auto* feat2 = reinterpret_cast<float*>(input2);

         // 2. 转换为OpenCV矩阵
         Mat vec1(1, 512, CV_32F, feat1);
         Mat vec2(1, 512, CV_32F, feat2);

         // 3. 关键：L2归一化（若原特征未归一化，必须添加这一步）
         normalize(vec1, vec1, 1.0, 0.0, NORM_L2);  // 归一化到模长为1
         normalize(vec2, vec2, 1.0, 0.0, NORM_L2);
         return norm(vec1 - vec2);
     }
}