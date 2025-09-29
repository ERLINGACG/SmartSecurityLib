//
// Created by HP on 2025/11/12.
//
#include "smartsecurity/cv/cvexport.h"
using cv_dnn::DnnYoloV5;
extern "C" {

    EXPORT_USE CORE_CV_API DnnYoloV5* DnnYoloV5Create(const char* path){
        return new DnnYoloV5(path);
    }
    EXPORT_USE CORE_CV_API void DnnYoloV5Destroy(const DnnYoloV5* dnnYoloV5){
        delete dnnYoloV5;
    }

}


extern "C" {
     EXPORT_USE CORE_CV_API void DnnYoloV5Debug(DnnYoloV5* dnnYoloV5){
        dnnYoloV5->Debug();
    }
    EXPORT_USE CORE_CV_API void DnnYoloV5DebugTime(DnnYoloV5* dnnYoloV5,int size,unsigned char* img,
        ImageData& OutputData,OutputJson& json
    )
     {
         dnnYoloV5->DebugTime(size,img,OutputData,json);
     }
    EXPORT_USE CORE_CV_API void DnnYoloV5Detection(DnnYoloV5* dnnYoloV5,int size,unsigned char* img,
        ImageData& OutputData,OutputJson& json
    )
     {
         dnnYoloV5->Detection(size,img,OutputData,json);
     }
}