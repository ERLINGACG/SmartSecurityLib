#ifndef SMART_SECURITY_FACIAL_RECOGNITION_H  // 修改为合法标识符
#define SMART_SECURITY_FACIAL_RECOGNITION_H
#include "smartsecurity/test/dlib/dlibexport.h"
#include "smartsecurity/test/dlib/data/output.h"
#include "smartsecurity/test/dlib/param/faceNew.h"

namespace dlib_face{
        class FacialRecognition {
            private:
               anet_type net;
               shape_predictor sp;
            public:
                explicit FacialRecognition(param::FaceNew* param);

                ~FacialRecognition()=default;


              void getDetection(
                unsigned char* inputData_1,
                int length,
                data::Output& outputData
              );

             static  dlib::matrix<float,0,1> byte_forMat(
                  unsigned char* inputData_1,
                  int length
              );

              static  double getDistance(
                 unsigned char* inputData_1,
                 int length_1,
                 unsigned char* inputData_2,
                 int length_2
              );
        };
      }

#endif