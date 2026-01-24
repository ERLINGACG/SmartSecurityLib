//
// Created by HP on 2025/12/9.
//
#include "smartsecurity/cv/cvexport.h"
#include "smartsecurity/cv/dnn/dnnYoloBoot.h"
using cv_dnn::DnnYoloBoot;
extern "C"{
     CORE_CV_API DnnYoloBoot* DnnYoloBootCreate(const char* path)
     {
         return new DnnYoloBoot(path);
     }

    CORE_CV_API void DnnYoloBootDestroy(const DnnYoloBoot* obj)
    {
        delete obj;
    }

}

extern "C" {
    EXPORT_USE CORE_CV_API void DnnYoloBootDetection(DnnYoloBoot* dnnYoloBoot,int size,unsigned char* img,
      ImageData& OutputData,OutputJson& json
    )
    {
        dnnYoloBoot->Detection(size,img,OutputData,json);
    }
}

