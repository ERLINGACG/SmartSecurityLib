//
// Created by HP on 2025/11/19.
//

#ifndef DNN_ARCFACE_H
#define DNN_ARCFACE_H
#include "dnnFaceNet.h"
using cv::Point2f;
using std::vector;
namespace cv_dnn {

    class DnnArcFace : public DnnFaceNet {
        vector<Point2f> standardLandmarks = {
            Point2f(38.2946f, 51.6963f),   // 右眼
            Point2f(73.5318f, 51.5014f),   // 左眼
            Point2f(56.0252f, 71.7366f),   // 鼻子
            Point2f(41.5493f, 92.3655f),   // 右嘴角
            Point2f(70.7299f, 92.2041f)    // 左嘴角
        };
    public:
        explicit DnnArcFace(const char* path);
        ~DnnArcFace() override=default;
        void Process(define::BasicDnnParams& params) override;
        void ProcessA(define::BasicDnnParams& params) ;
        double ArcVerifyFeature512(unsigned char* input1,unsigned char* input2) {
            logger.setOwnerFuncName(__func__);
            auto* feat1 = reinterpret_cast<float*>(input1);
            auto* feat2 = reinterpret_cast<float*>(input2);

            Mat vec1(1, 512, CV_32F, feat1);
            Mat vec2(1, 512, CV_32F, feat2);
            cv::normalize(vec1, vec1, 1.0, 0.0, cv::NORM_L2);  // 归一化到模长为1
            cv::normalize(vec2, vec2, 1.0, 0.0, cv::NORM_L2);
            logger.info("distance:"+std::to_string(cv::norm(vec1 - vec2)));
            return cv::norm(vec1 - vec2);
        }

    };

}
#endif //DNN_ARCFACE_H
