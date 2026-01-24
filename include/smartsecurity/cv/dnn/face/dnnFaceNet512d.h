//
// Created by HP on 2025/11/17.
//

#ifndef DNN_FACENET_512D_H
#define DNN_FACENET_512D_H
#include "dnnFaceNet.h"

namespace cv_dnn {
        class DnnFaceNet512d : public DnnFaceNet {
        public:
            explicit DnnFaceNet512d(const char *path);
            ~DnnFaceNet512d() override =default;

            void Process(define::BasicDnnParams& params) override;


            double VerifyFeature512d(unsigned char* input1,unsigned char* input2){
                logger.setOwnerFuncName(__func__);

                // 1. 将unsigned char指针转为float特征向量
                auto* feat1 = reinterpret_cast<float*>(input1);
                auto* feat2 = reinterpret_cast<float*>(input2);

                // 2. 转换为OpenCV矩阵
                Mat vec1(1, 512, CV_32F, feat1);
                Mat vec2(1, 512, CV_32F, feat2);

                // 3. 关键：L2归一化（若原特征未归一化，必须添加这一步）
                cv::normalize(vec1, vec1, 1.0, 0.0, cv::NORM_L2);  // 归一化到模长为1
                cv::normalize(vec2, vec2, 1.0, 0.0, cv::NORM_L2);
                return cv::norm(vec1 - vec2);
            }
        };
}
#endif //DNN_FACENET_512D_H
