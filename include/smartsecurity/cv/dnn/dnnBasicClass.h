//
// Created by HP on 2025/8/15.
//

#ifndef SMARTSECURITYCORELIB_DNN_BASIC_CLASS_H
#define SMARTSECURITYCORELIB_DNN_BASIC_CLASS_H
#include <opencv2/dnn.hpp>
#include "opencv2/mcc/checker_model.hpp"
#include "smartsecurity/cv/dnn/dnnInterface.h"

namespace cv_dnn::dnnBasic{
class DnnBasicClass:public cv_dnn::interface::DnnInterface{
    public:
       cv::dnn::Net net;
    public:
        ~DnnBasicClass() override =default;
        void DetectImage(unsigned char* inputData,int size,data::ImageData& OutputData) override =0;

    };
}
#endif //SMARTSECURITYCORELIB_DNN_BASIC_CLASS_H
