//
// Created by HP on 2025/12/9.
//
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "smartsecurity/cv/dnn/dnnYoloBoot.h"
#include "nlohmann/json.hpp"

cv_dnn::DnnYoloBoot::DnnYoloBoot(const char* path)
{
    logger.setOwnerName("DnnYoloBoot");
    this->Load(path);
    this->DnnYoloBoot::InitConfig();
}
void cv_dnn::DnnYoloBoot::InitConfig()
{
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
    this->net= cv::dnn::readNetFromONNX(string(_config["modelPath"]));
    this->isEnableFusion=_config["isEnableFusion"];
    this->net.enableFusion(isEnableFusion);
    if (net.empty()){
        logger.error("load model failed");
        return;
    }
    if (_config["isCUDA"]) {
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        logger.info("use CUDA");
    }else{
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        logger.info("use CPU");
    }
    logger.info("init success");
}

void cv_dnn::DnnYoloBoot::DebugTime(int size, unsigned char* img, ImageData& OutputData, OutputJson& json)
{
}

void cv_dnn::DnnYoloBoot::PacketData(define::BasicDnnParams& params, Mat& input)
{
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<YoloPrams&>(params);
    dnnParams.input=input;
}

void cv_dnn::DnnYoloBoot::UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson)
{
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<YoloPrams&>(params);
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



void cv_dnn::DnnYoloBoot::Input(int sizes, unsigned char* img, Mat& output)
{
    logger.setOwnerFuncName(__func__);
    const std::vector buf(img, img + sizes); //构造临时缓冲区
    output = imdecode(buf, cv::IMREAD_COLOR);  // 修正参数为正确解码标志
}

void cv_dnn::DnnYoloBoot::SetBlob(define::BasicDnnParams& params)
{
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<YoloPrams&>(params);
    dnnParams.blob= cv::dnn::blobFromImage(
        dnnParams.input,
        1.0 / 255.0,
        cv::Size(this->inputWidth, this->inputHeight),
        cv::Scalar(0, 0, 0),
        true,
        false
    );
    this->net.setInput(dnnParams.blob);
}

void cv_dnn::DnnYoloBoot::Forward(define::BasicDnnParams& params)
{
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<YoloPrams&>(params);
    dnnParams.forward_output = this->net.forward( "output0");
}

void cv_dnn::DnnYoloBoot::Process(define::BasicDnnParams& params)
{
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<YoloPrams&>(params);
    int numChannels = dnnParams.forward_output.size[1];  // 84（4坐标 + 80类别）
    int numAnchors = dnnParams.forward_output.size[2];   // 8400（候选框数量）
    // std::cout << "numChannels = " << numChannels << std::endl;
    // std::cout << "numAnchors = " << numAnchors << std::endl;
    auto data = dnnParams.forward_output.ptr<float>(0);
    float x_factor = static_cast<float>(dnnParams.input.cols) / static_cast<float>(this->inputWidth);
    float y_factor = static_cast<float>(dnnParams.input.rows) / static_cast<float>(this->inputHeight);
    for(int w = 0; w < numAnchors; w++)
    {
        float maxScore = 0.0f;
        int classId = -1;
        for (int h = 4; h < numChannels; h++)
        {
            float score = data[h * numAnchors + w];
            if (score > maxScore) {
                maxScore = score;
                classId = h - 4;
            }

        }
        if (maxScore > 0.5)
        {
            const float cx = data[0 * numAnchors + w];  // 中心 x
            const float cy = data[1 * numAnchors + w];  // 中心 y
            const float w_box = data[2 * numAnchors + w];// 框宽
            const float h_box = data[3 * numAnchors + w];// 框高
            const int x1 = static_cast<int>((cx - w_box / 2) * x_factor);
            const int y1 = static_cast<int>((cy - h_box / 2) * y_factor);
            const int x2 = static_cast<int>((cx + w_box / 2) * x_factor);
            const int y2 = static_cast<int>((cy + h_box / 2) * y_factor);
            dnnParams.boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            dnnParams.classIds.emplace_back(classId);
            dnnParams.confidences.emplace_back(maxScore);
        }

    }
    cv::dnn::NMSBoxes(
       dnnParams.boxes,
       dnnParams.confidences,
       static_cast<float>(this->confThreshold),
       static_cast<float>(this->nmsThreshold),
       dnnParams.indices
   );

}

void cv_dnn::DnnYoloBoot::PostProcessing(define::BasicDnnParams& params)
{
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<YoloPrams&>(params);
    auto dataMap = std::map<std::string,int>();
    for (auto i:dnnParams.indices)
    {
        const auto box = (dnnParams.boxes)[i];
        const auto confidences = (dnnParams.confidences)[i];
        rectangle(dnnParams.input, box, cv::Scalar(255,0,0), 2);
        putText(dnnParams.input,
            this->classNames[dnnParams.classIds[i]] + ": " + to_string(confidences),
            cv::Point(box.x, box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5, cv::Scalar(0, 255, 0), 2
            );
        dataMap[this->classNames[dnnParams.classIds[i]]]++;
    }
    auto makeJson=[&]()
    {
        string  dataJson;
        if (!dataMap.empty()) {

           const json json=dataMap;
           dataJson=json.dump();
           auto buffer = std::make_unique<char[]>(dataJson.size() + 1);
           std::ranges::copy(dataJson, buffer.get());
           buffer[dataJson.size()] = '\0';
           dnnParams.outputJson.json = std::move(buffer);
           dnnParams.outputJson.size = static_cast<int>(dataJson.size());

        }else {
            dataJson.clear();
        }
    };
    makeJson();
}

void cv_dnn::DnnYoloBoot::Detection(int size, unsigned char* img, ImageData& OutputData, OutputJson& json)
{
    logger.setOwnerFuncName(__func__);
    Mat input;
    const auto dnnParams=YoloPrams::Create();
    logger.time([&](){this->Input(size,img,input);});
    logger.time([&](){this->PacketData(*dnnParams,input);});
    logger.time([&](){this->SetBlob(*dnnParams);});
    logger.time([&](){this->Forward(*dnnParams);});
    logger.time([&](){this->Process(*dnnParams);});
    logger.time([&](){this->PostProcessing(*dnnParams);});
    logger.time([&](){this->UnPacketData(*dnnParams,OutputData,json);});
}

void cv_dnn::DnnYoloBoot::SetSR(cv_sr::DnnSR* sr)
{

}

void cv_dnn::DnnYoloBoot::Upsample(Mat& input, Mat& output)
{

}

void cv_dnn::DnnYoloBoot::DebugTimeSR(int size, unsigned char* img, ImageData& OutputData, OutputJson& json)
{
}



