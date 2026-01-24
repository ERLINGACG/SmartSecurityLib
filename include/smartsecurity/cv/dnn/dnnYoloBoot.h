//
// Created by HP on 2025/12/9.
//

#ifndef DNN_YOLO_BOOT_H
#define DNN_YOLO_BOOT_H
#include "dnnYoloV5.h"

namespace cv_dnn
{

        struct YoloPrams: define::BasicDnnParams
        {
            Mat input;       //输入图像
            Mat blob;        //blob图像
            Mat forward_output; //前向传播输出
            vector<Rect> boxes; //检测框
            vector<float> confidences; //置信度
            vector<int> classIds; //类别ID
            vector<int> indices; //NMS索引
            OutputJson      outputJson;

            static unique_ptr<YoloPrams> Create() {
                return std::make_unique<YoloPrams>();
            }
        };
        class DnnYoloBoot : public dnnNetFunc, public dnnBasic::DnnBasicClass{
            double confThreshold{};
            double nmsThreshold{};
            int    inputWidth{};
            int    inputHeight{};
            bool   isEnableFusion=false;
            cv_sr::DnnSR* sr{};
            std::vector<string> classNames={};
        public:
            explicit DnnYoloBoot(const char* path);
            ~DnnYoloBoot() override=default ;


            void DebugTime(int size,unsigned char* img,ImageData& OutputData,OutputJson& json);
            void PacketData(define::BasicDnnParams& params,   Mat& input) override; //封包
            void UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson) override; //解包
            void InitConfig() override;
            void Input(int sizes,unsigned char* img,Mat& output) override;
            void SetBlob(define::BasicDnnParams& params) override;
            void Forward(define::BasicDnnParams& params) override;
            void Process(define::BasicDnnParams& params) override;
            void PostProcessing(define::BasicDnnParams& params) override;
            void Detection(int size,unsigned char* img,ImageData& OutputData,OutputJson& json) override;
            void SetSR(cv_sr::DnnSR* sr);
            void Upsample(Mat& input,Mat& output);
            void DebugTimeSR(int size,unsigned char* img,ImageData& OutputData,OutputJson& json);


        };
}
#endif //DNN_YOLO_BOOT_H
