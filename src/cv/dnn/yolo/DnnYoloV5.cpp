//
// Created by HP on 2025/11/12.
//
#include "opencv2/imgcodecs.hpp"
#include "smartsecurity/cv/dnn/dnnYoloV5.h"
#include <sstream>
#include <string>

#include "opencv2/imgproc.hpp"

using cv_dnn::DnnYoloV5;

using cv::Size;
using cv::MatSize;
using cv::Point;
using cv::dnn::readNetFromONNX;
using cv::dnn::DNN_BACKEND_CUDA;
using cv::dnn::DNN_TARGET_CUDA;
using cv::dnn::DNN_BACKEND_OPENCV;
using cv::dnn::DNN_TARGET_CPU;
using cv::dnn::blobFromImage;
using cv::dnn::NMSBoxes;
using cv::imdecode;
using cv::putText;
using cv::rectangle;
using std::map;


DnnYoloV5::DnnYoloV5(const char* path): confThreshold(0), nmsThreshold(0) {

    logger.setOwnerName("DnnYoloV5");
    this->Load(path);
    this->DnnYoloV5::InitConfig();
}

void DnnYoloV5::Debug() {
    auto className=[this]
    {
      string result;
      for (const auto& e : classNames){
        result += e + ",";
      }
      return "["+result+"]";
    };
    logger.setOwnerFuncName("Debug");
    logger.info(to_string(confThreshold));
    logger.info(to_string(nmsThreshold));
    logger.info(to_string(this->inputWidth));
    logger.info(to_string(this->inputHeight));
    logger.info(to_string(this->classNames.size()));
    logger.info(className());


}

void DnnYoloV5::DebugTime(int size,unsigned char* img,ImageData& OutputData,OutputJson& json)
{
    Mat input;
    auto test=new IDnnParamsV5();
    logger.time(&DnnYoloV5::Input,this,size,img,input);
    logger.time(&DnnYoloV5::PacketData,this,*test,input);
    logger.time(&DnnYoloV5::SetBlob,this,*test);
    logger.time(&DnnYoloV5::Forward,this,*test);
    logger.time(&DnnYoloV5::Process,this,*test);
    logger.time(&DnnYoloV5::PostProcessing,this,*test);
    logger.time(&DnnYoloV5::UnPacketData,this,*test,OutputData,json);
}

void DnnYoloV5::SetSR(cv_sr::DnnSR* sr)
{
    this->sr=sr;
}

void DnnYoloV5::Upsample(Mat& input, Mat& output)
{
    logger.setOwnerFuncName(__func__);
    this->sr->Upsample(input,output);
}

void DnnYoloV5::DebugTimeSR(int size, unsigned char* img, ImageData& OutputData, OutputJson& json)
{
    Mat input;
    auto test=IDnnParamsV5::Create();
    logger.time(&DnnYoloV5::Input,this,size,img,input);
    // if (input.rows< 320 || input.cols < 240)
    // {
        logger.time(&DnnYoloV5::Upsample,this,input,input);
    // }
    
    logger.time(&DnnYoloV5::PacketData,this,*test,input);
    logger.time(&DnnYoloV5::SetBlob,this,*test);
    logger.time(&DnnYoloV5::Forward,this,*test);
    logger.time(&DnnYoloV5::Process,this,*test);
    logger.time(&DnnYoloV5::PostProcessing,this,*test);
    logger.time(&DnnYoloV5::UnPacketData,this,*test,OutputData,json);

}


void DnnYoloV5::InitConfig() {
    logger.setOwnerFuncName(__func__);
    string env   =config["env"];
    json _config =config[env];
    this->nmsThreshold  =_config["nmsThreshold"];
    this->confThreshold =_config["confThreshold"];
    this->inputWidth    =_config["inputWidth"];
    this->inputHeight   =_config["inputHeight"];
    for (const auto& e : _config["classNames"]){
        this->classNames.emplace_back(e);
    }
    this->net=readNetFromONNX(string(_config["modelPath"]));
    if (net.empty()){
        logger.error("load model failed");
        return;
    }
    if (_config["isCUDA"]) {
        this->net.setPreferableBackend(DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(DNN_TARGET_CUDA);
        logger.info("use CUDA");
    }else{
        this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
        this->net.setPreferableTarget(DNN_TARGET_CPU);
        logger.info("use CPU");
    }
    logger.info("init success");
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
void DnnYoloV5::PacketData(define::BasicDnnParams& params,Mat& input) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsV5&>(params);
    dnnParams.input=input;
}

void DnnYoloV5::UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsV5&>(params);
    vector params_ {cv::IMWRITE_WEBP_QUALITY, 95};
    vector<unsigned char> encoded;
    imencode(".jpeg", dnnParams.input, encoded, params_);
    imageData.size =  static_cast<int>(encoded.size());
    imageData.width = dnnParams.input.cols;
    imageData.height = dnnParams.input.rows;
    imageData.channels = 3;
    imageData.data = std::make_unique<unsigned char[]>(encoded.size());
    memcpy(imageData.data.get(), encoded.data(), encoded.size());
    outputJson = std::move(dnnParams.outputJson);  // 移动 OutputJson
}

void DnnYoloV5::Input(int sizes, unsigned char* img, Mat& output) {
    logger.setOwnerFuncName(__func__);
    if(img==nullptr || sizes<=0){
        logger.error("inputImage failed,img is null or size is 0");
        return;
    }
    const std::vector buf(img, img + sizes); //构造临时缓冲区
    output = imdecode(buf, cv::IMREAD_COLOR);  // 修正参数为正确解码标志
}

void DnnYoloV5::SetBlob(define::BasicDnnParams& params) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsV5&>(params);
    dnnParams.blob=blobFromImage(
        dnnParams.input,
        1.0 / 255.0,
        Size(this->inputWidth, this->inputHeight),
        cv::Scalar(0, 0, 0),
        true,
        false
    );
    this->net.setInput(dnnParams.blob);
}

void DnnYoloV5::Forward(define::BasicDnnParams& params) {

    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsV5&>(params);

    auto Ito_string =[](MatSize size){
        std::ostringstream oss;
        oss << size;  // 利用OpenCV重载的operator<<，直接输出维度信息
        return oss.str();
    };

    vector<string> outputs_name = this->net.getUnconnectedOutLayersNames(); // 获取输出层名称
    this->net.forward(dnnParams.forward,outputs_name); // 前向传播

    dnnParams.matrixSize=Ito_string(dnnParams.forward[0].size);
}

void DnnYoloV5::Process(define::BasicDnnParams& params) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsV5&>(params);

    auto* data     = reinterpret_cast<float*>(dnnParams.forward[0].data);
    float x_factor = static_cast<float>(dnnParams.input.cols) / 640.0f;
    float y_factor = static_cast<float>(dnnParams.input.rows) / 640.0f;

    int rows = 25200;
    for (int i = 0; i < rows; i++) {

        float confidence = data[4];
        if (confidence > this->confThreshold)
        {
            float* classes_scores = data + 5;    //分类分数
            Mat scores(1, static_cast<int>(this->classNames.size()), CV_32FC1, classes_scores); //转换为Mat
            Point class_id;     //类别索引
            double max_class_score; //最大分数
            minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id); //找到最大分数的索引
            if (max_class_score > 0.25) {
                dnnParams.confidences.push_back(confidence);//置信度
                dnnParams.classIds.push_back(class_id.x);  //类别索引
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left   = static_cast<int>((x - 0.5 * w) * x_factor); //左上角x坐标
                int top    = static_cast<int>((y - 0.5 * h) * y_factor); //左上角y坐标
                int width  = static_cast<int>(w * x_factor);           //宽度
                int height = static_cast<int>(h * y_factor);          //高度
                dnnParams.boxes.emplace_back(left, top, width, height);
            }
        }
        data += this->classNames.size()+5; //跳过85个元素，到下一个检测框
    }

    NMSBoxes(
               dnnParams.boxes, dnnParams.confidences,
               static_cast<float>(this->confThreshold),
               static_cast<float>(this->nmsThreshold), dnnParams.nmsResult
    ); //非极大值抑制
    dnnParams.boxSize=to_string(dnnParams.boxes.size());
}

void DnnYoloV5::PostProcessing(define::BasicDnnParams& params) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsV5&>(params);
    auto dataMap = map<std::string,int>();
    for(int index : dnnParams.nmsResult){
        Rect box = dnnParams.boxes[index];
        if(this->classNames[dnnParams.classIds[index]]!="dining table"){

            rectangle(dnnParams.input, box, cv::Scalar(0, 255, 0), 2);
            putText(dnnParams.input, this->classNames[dnnParams.classIds[index]] + ": " + to_string(dnnParams.confidences[index]), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            // 直接增加计数，如果键不存在会自动创建并初始化为0
            dataMap[this->classNames[dnnParams.classIds[index]]]++;
        }
    }
    auto Json =[&]()
    {
        string  dataJson;
        if(!dataMap.empty()){

            for (const auto& pair : dataMap) {
                dataJson += "\"" + pair.first + "\": " + std::to_string(pair.second) + ",";
            }
            dataJson.pop_back(); // 移除最后一个逗号
            dataJson = "{" + dataJson + "}";
            auto buffer = std::make_unique<char[]>(dataJson.size() + 1);
            std::ranges::copy(dataJson, buffer.get());
            buffer[dataJson.size()] = '\0';
            dnnParams.outputJson.json = std::move(buffer);
            dnnParams.outputJson.size = static_cast<int>(dataJson.size());
        }else{
            dataJson.clear();
        }
    };
    Json();

}


// ReSharper disable once CppParameterMayBeConst
void DnnYoloV5::Detection(int size,unsigned char* img,ImageData& OutputData,OutputJson& json) {
    logger.setOwnerFuncName(__func__);
    Mat input;
    auto dnnParams=IDnnParamsV5::Create();
    this->Input(size,img,input);
    this->PacketData(*dnnParams,input);
    this->SetBlob(*dnnParams);
    this->Forward(*dnnParams);
    this->Process(*dnnParams);
    this->PostProcessing(*dnnParams);
    this->UnPacketData(*dnnParams,OutputData,json);
}
