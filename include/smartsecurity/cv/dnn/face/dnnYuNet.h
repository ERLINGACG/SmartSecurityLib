//
// Created by HP on 2025/11/20.
//

#ifndef DNN_YUNET_H
#define DNN_YUNET_H
#include "smartsecurity/cv/dnn/dnnBasicClass.h"
#include "opencv2/face.hpp"
#include <opencv2/objdetect/face.hpp>

#include "dnnArcFace.h"
#include "dlib/geometry/vector.h"
#include "smartsecurity/cv/dnn/dnnNetFunc.h"
using cv::Ptr;
using cv::FaceDetectorYN;
using std::unique_ptr;
using std::vector;
using cv::Rect;
using data::FaceFeatureByte;
namespace cv_dnn {

    struct IDnnParamsYuNet : define:: BasicDnnParams{
        int              count=0;
        Mat              input;
        Mat              faces;
        vector<Rect>     facesRect;
        vector<vector<float>> faceFeatures;
        FaceFeatureByte       faceFeatureByte;

        static unique_ptr<IDnnParamsYuNet> Create() {
            return std::make_unique<IDnnParamsYuNet>();
        }
    };

    class DnnYuNet : public dnnBasic::DnnBasicClass,public dnnNetFunc{

        Ptr<FaceDetectorYN> detector;
        int inputWidth{};
        int inputHeight{};
        int maxFace{};
        float scoreThreshold{};
        float nmsThreshold{};
        DnnArcFace* arcFace{};
        public:
            explicit DnnYuNet(const char* configPath);

            ~DnnYuNet() override =default;

            void InitConfig() override;
            void getInfo();
            void setArcFace(DnnArcFace* arc);
            void DebugTime(int size, unsigned char* img, ImageData& OutputData, OutputJson& json,FaceFeatureByte& faceFeatureByte);

            void PacketData(define::BasicDnnParams& params, Mat& input) override;
            void UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson) override;
            void UnPacketFeature(define::BasicDnnParams& params, FaceFeatureByte& faceFeatureByte);
            void Input(int sizes, unsigned char* img, Mat& output) override;
            void SetBlob(define::BasicDnnParams& params) override;
            void Forward(define::BasicDnnParams& params) override{};
            void Process(define::BasicDnnParams& params) override;
            void PostProcessing(define::BasicDnnParams& params) override;
            void Detection(int size, unsigned char* img, ImageData& OutputData, OutputJson& json) override{};
    };

}
#endif //DNN_YUNET_H
