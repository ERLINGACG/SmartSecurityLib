//
// Created by HP on 2025/10/14.
//

#include <smartsecurity/cv/cvexport.h>
using data::ImageData;
using data::FaceFeatureByte;
using data::json::OutputJson;
extern "C"{
    EXPORT_USE CORE_CV_API void ImageDataDestroy  (const ImageData* i)      { delete i; }
    EXPORT_USE CORE_CV_API void FaceFeatureDestroy(const FaceFeatureByte* i){ delete i; }
    EXPORT_USE CORE_CV_API void OutputJsonDestroy (const OutputJson* o)     { delete o; }
}
