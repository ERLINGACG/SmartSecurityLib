//
// Created by HP on 2025/11/12.
//

#ifndef DNN_YOLOV5_H
#define DNN_YOLOV5_H
#include <memory>
#include "dnnBasicClass.h"
#include "dnnNetFunc.h"
#include "smartsecurity/data/image_data.h"
#include "smartsecurity/data/output_json.h"
#include "sr/dnnsr.h"
using cv::Mat;
using cv::Rect;
using std::vector;
using data::ImageData;
using data::json::OutputJson;
using std::unique_ptr;
namespace cv_dnn
{

    struct IDnnParamsV5 : define::BasicDnnParams{
        Mat             input;
        Mat             blob;
        string          matrixSize;
        string          boxSize;
        vector<Mat>     forward;
        vector<int>     classIds;
        vector<float>   confidences;
        vector<Rect>    boxes;
        vector<int>     nmsResult;
        OutputJson      outputJson;


        static unique_ptr<IDnnParamsV5> Create() {
            return std::make_unique<IDnnParamsV5>();
        }

    };
    class DnnYoloV5 : public dnnNetFunc, public dnnBasic::DnnBasicClass{

        double confThreshold;
        double nmsThreshold;
        int    inputWidth{};
        int    inputHeight{};
        cv_sr::DnnSR* sr{};
        std::vector<string> classNames={};
        public:

                 explicit DnnYoloV5(const char* path);
                ~DnnYoloV5() override=default ;

                void Debug();
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
#endif //DNN_YOLOV5_H
