//
// Created by HP on 2025/11/12.
//

#ifndef DNN_NET_INF_H
#define DNN_NET_INF_H
#include "smartsecurity/data/image_data.h"
#include "smartsecurity/data/output_json.h"
using cv::Mat;
using data::ImageData;
using data::json::OutputJson;
namespace cv_dnn
{
    namespace define {
        struct BasicDnnParams
        {
            virtual ~BasicDnnParams() = default;
        };
    }


    class dnnNetFunc
    {
    public:

        virtual ~dnnNetFunc() = default;
        virtual void  InitConfig()=0;
        virtual void  PacketData(define::BasicDnnParams& params,   Mat& input)=0; //封包
        virtual void  UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson)=0; //解包
        virtual void  Input(int sizes,unsigned char* img,Mat& output)=0;
        virtual void  SetBlob(define::BasicDnnParams& params)=0;
        virtual void  Forward(define::BasicDnnParams& params)=0;
        virtual void  Process(define::BasicDnnParams& params)=0;
        virtual void  PostProcessing(define::BasicDnnParams& params)=0;
        virtual void  Detection(int size,unsigned char* img,ImageData& OutputData,OutputJson& json)=0;


    };
}
#endif //DNN_NET_INF_H
