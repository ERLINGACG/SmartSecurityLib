//
// Created by HP on 2025/11/14.
//

#ifndef DNN_SSD_CAFFEM_H
#define DNN_SSD_CAFFEM_H
#include "../dnnBasicClass.h"
#include "dnnFaceNet.h"
#include "../dnnNetFunc.h"
// #include "dnnYoloV5.h"
using cv::Mat;
using std::unique_ptr;
using std::vector;
using cv::Rect;
using std::map;
using data::FaceFeatureByte;
namespace cv_dnn{
     struct IDnnParamsSSDcaffem: define::BasicDnnParams{
         int              count=0;
         Mat              input;
         Mat              blob;
         Mat              forward;
         vector<Rect>     faces;
         vector<float>    confidences;
         map<int,float>   message;
         vector<vector<float>> faceFeatures;
         FaceFeatureByte       faceFeatureByte;
         OutputJson            outputJson;

         static unique_ptr<IDnnParamsSSDcaffem> Create(){
             return std::make_unique<IDnnParamsSSDcaffem>();
         }
     };

     class DnnSSDcaffem: public dnnNetFunc, public dnnBasic::DnnBasicClass{
         int inputWidth{};
         int inputHeight{};
         double confThreshold{};

     public:
         DnnFaceNet* faceNet;
          explicit DnnSSDcaffem(const char* path,DnnFaceNet* faceNet);
         ~DnnSSDcaffem() override =default;

         void DebugTime(int size,unsigned char* img,ImageData& OutputData,OutputJson& json);
         void InitConfig() override;
         void PacketData(define::BasicDnnParams& params,   Mat& input) override; //封包
         void UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson) override; //解包
         void Input(int sizes,unsigned char* img,Mat& output) override;
         void SetBlob(define::BasicDnnParams& params) override;
         void Forward(define::BasicDnnParams& params) override;
         void Process(define::BasicDnnParams& params) override;
         void PostProcessing(define::BasicDnnParams& params) override;
         void Detection(int size,unsigned char* img,ImageData& OutputData,OutputJson& json) override;
         void DetectionFeature(int size,unsigned char* img,ImageData& OutputData,OutputJson& json,FaceFeatureByte& faceFeatureByte);
         void DetectionFeature512d(int size,unsigned char* img,ImageData& OutputData,OutputJson& json,FaceFeatureByte& faceFeatureByte);
         void DetectionFeatureArc(int size,unsigned char* img,ImageData& OutputData,OutputJson& json,FaceFeatureByte& faceFeatureByte);
     };
}
#endif //DNN_SSD_CAFFEM_H
