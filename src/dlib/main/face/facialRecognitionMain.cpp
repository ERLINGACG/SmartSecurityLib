#include "smartsecurity/test/dlib/dlibexport.h"


extern "C"{
    CORE_DLIB_API dlib_face::FacialRecognition* createFacialRecognition(param::FaceNew* param){
        return new dlib_face::FacialRecognition(param);
    }
    CORE_DLIB_API void deleteFacialRecognition(dlib_face::FacialRecognition* facialRecognition){
        delete facialRecognition;
    }
} 
extern "C" {
    CORE_DLIB_API void getDetection(
            dlib_face::FacialRecognition* facialRecognition,
            unsigned char* img,
            int length, data::Output& outputData
    ){
        facialRecognition->getDetection(img, length, outputData);
    }
    CORE_DLIB_API double getDistance(dlib_face::FacialRecognition* facialRecognition,
                                     unsigned char* byte1,
                                     int length1,
                                     unsigned char* byte2,
                                     int length2
    ){
       return dlib_face::FacialRecognition::getDistance(
                 byte1,length1,byte2,length2
                 );
    }
}
extern  "C"{
     CORE_DLIB_API int ClearOutputData(data::Output& outputData){
        return dlib_utils::DlibUtils::clearOutputData(outputData);
     }
     CORE_DLIB_API int DeleteOutputData(data::Output* outputData){
        return dlib_utils::DlibUtils::deleteOutputData(outputData);
     }
}