#include <iostream>
#include <mutex>
#include "smartsecurity/cv/cvexport.h"

using namespace dnnDetection;
using namespace cv_dnn;
extern "C"{
    CORE_CV_API DnnDetectorYolo* createDnnDetector(const char* path, double confidence, double nms){
        return new DnnDetectorYolo(path, true, confidence, nms);
    }
    CORE_CV_API DnnDetectorYolo* createDnnDetector_1(const char* jsonPath){
        return new DnnDetectorYolo(jsonPath);
    }
    CORE_CV_API void destroyDnnDetector(DnnDetectorYolo* detector){
        delete detector;
    }
}

extern "C"{
    CORE_CV_API void DnnDetectorYolo(DnnDetectorYolo* detector,
                                     unsigned char* inputData, int size,
                                     data::ImageData& OutputData,data::json::OutputJson& json)
    {

        try{
            detector->DetectImage_3(inputData,size,OutputData,json);
        }catch(std::exception& e){
            std::cout << e.what() << std::endl;
        }
    }

}