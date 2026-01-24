//
// Created by HP on 2025/11/14.
//

#ifndef DNN_FACENET_H
#define DNN_FACENET_H
#include "../dnnBasicClass.h"
#include "../dnnNetFunc.h"
using data::FaceFeatureByte;
namespace cv_dnn {
    class DnnFaceNet :public dnnNetFunc, public dnnBasic::DnnBasicClass{
    public:
         int inputWidth{},inputHeight{};

         explicit DnnFaceNet(const char *path);
        ~DnnFaceNet() override =default;
        void InitConfig() override;
        void PacketData(define::BasicDnnParams& params, Mat& input) override{};
        void UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson) override{};
        void Input(int sizes, unsigned char* img, Mat& output) override{};
        void SetBlob(define::BasicDnnParams& params) override{};
        void Forward(define::BasicDnnParams& params) override{};
        void Process(define::BasicDnnParams& params) override;
        void PostProcessing(define::BasicDnnParams& params) override;
        void Detection(int size, unsigned char* img, ImageData& OutputData, OutputJson& json) override{};

        void UnPacketFeature(define::BasicDnnParams& params, FaceFeatureByte& faceFeatureByte);

        double VerifyFeature(unsigned char* input1,unsigned char* input2) {
            logger.setOwnerFuncName(__func__);
            auto* feat1 = reinterpret_cast<float*>(input1);
            auto* feat2 = reinterpret_cast<float*>(input2);

            Mat vec1(1, 128, CV_32F, feat1);
            Mat vec2(1, 128, CV_32F, feat2);
            cv::normalize(vec1, vec1, 1.0, 0.0, cv::NORM_L2);  // 归一化到模长为1
            cv::normalize(vec2, vec2, 1.0, 0.0, cv::NORM_L2);
            return cv::norm(vec1 - vec2);
        }
    };
}
#endif //DNN_FACENET_H
