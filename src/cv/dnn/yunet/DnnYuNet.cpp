//
// Created by HP on 2025/11/20.
//
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "smartsecurity/cv/dnn/face/dnnYuNet.h"

using cv_dnn::DnnYuNet;
using cv::Mat;
using std::vector;
using cv::imdecode;


DnnYuNet::DnnYuNet(const char* configPath) {
    this->logger.setOwnerName("DnnYuNet");
    Load(configPath);
    DnnYuNet::InitConfig();
}

void DnnYuNet::InitConfig() {
      logger.setOwnerFuncName(__func__);
      string env=config["env"];
      json runtime = config[env];
      this->inputHeight=runtime["inputHeight"];
      this->inputWidth=runtime["inputWidth"];
     this->maxFace=runtime["maxFace"];
     this->scoreThreshold=runtime["scoreThreshold"];
     this->nmsThreshold=runtime["nmsThreshold"];
      if (runtime["isCUDA"]) {
          this->detector=FaceDetectorYN::create(runtime["yunet_path"].get<string>(),
                                            "",
                                             cv::Size(this->inputWidth,this->inputHeight),
                                             this->scoreThreshold,
                                             this->nmsThreshold,
                                             this->maxFace, //置信度，NMS阈值，最大检测数
                                             cv::dnn::DNN_BACKEND_CUDA,
                                             cv::dnn::DNN_TARGET_CUDA
                                             );
          logger.info("FaceDetectorYN create with CUDA");
      }else {
          this->detector=FaceDetectorYN::create(runtime["yunet_path"].get<string>(),
                                            "",
                                             cv::Size(this->inputWidth,this->inputHeight),
                                             this->scoreThreshold,
                                             this->nmsThreshold,
                                             this->maxFace //置信度，NMS阈值，最大检测数
                                   );
      }

     if (this->detector.empty()) {
         this->logger.error("FaceDetectorYN create failed");
     }else {
         logger.info("FaceDetectorYN create success");
     }

}

void DnnYuNet::getInfo(){
    logger.info("inputWidth:"+to_string(this->inputWidth));
    logger.info("inputHeight:"+to_string(this->inputHeight));
    logger.info("maxFace:"+to_string(this->maxFace));
    logger.info("scoreThreshold:"+to_string(this->scoreThreshold));
    logger.info("nmsThreshold:"+to_string(this->nmsThreshold));
    logger.info("model Load:"+to_string(!this->detector.empty()));
}

void DnnYuNet::setArcFace(DnnArcFace* arc)
{
    this->arcFace =arc;
}

void DnnYuNet::DebugTime(int size, unsigned char* img, ImageData& OutputData, OutputJson& json,FaceFeatureByte& faceFeatureByte)
{
    Mat inputImage;
    auto dnnParams=IDnnParamsYuNet::Create();
    logger.time(&DnnYuNet::Input,this,size,img,inputImage);
    logger.time(&DnnYuNet::PacketData,this,*dnnParams,inputImage);
    logger.time(&DnnYuNet::Process,this,*dnnParams);
    logger.time([&]
    {
        this->arcFace->ProcessA(*dnnParams);
    });
    logger.time(&DnnYuNet::PostProcessing,this,*dnnParams);
    logger.time(&DnnYuNet::UnPacketData,this,*dnnParams,OutputData,json);
    logger.time(&DnnYuNet::UnPacketFeature,this,*dnnParams,faceFeatureByte);


}

void DnnYuNet::PacketData(define::BasicDnnParams& params, Mat& input) {
    logger.setOwnerFuncName(__func__);
     if(input.empty()){
        logger.error("inputImage failed,input is empty");
        return;
    }
    auto& paramsYuNet=dynamic_cast<IDnnParamsYuNet&>(params);
    paramsYuNet.input=input;

}

void DnnYuNet::UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson) {

    logger.setOwnerFuncName(__func__);
    auto& paramsYuNet=dynamic_cast<IDnnParamsYuNet&>(params);
    vector params_ {cv::IMWRITE_WEBP_QUALITY, 95};
    vector<unsigned char> encoded;
    imencode(".jpeg", paramsYuNet.input, encoded, params_);
    imageData.size =  static_cast<int>(encoded.size());
    imageData.width = paramsYuNet.input.cols;
    imageData.height = paramsYuNet.input.rows;
    imageData.channels = 3;
    imageData.data = std::make_unique<unsigned char[]>(encoded.size());
    memcpy(imageData.data.get(), encoded.data(), encoded.size());
    // outputJson = std::move(dnnParams.outputJson);  // 移动 OutputJson
}

void DnnYuNet::UnPacketFeature(define::BasicDnnParams& params, FaceFeatureByte& faceFeatureByte) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsYuNet&>(params);
    faceFeatureByte=std::move(dnnParams.faceFeatureByte);
}

void DnnYuNet::Input(int sizes, unsigned char* img, Mat& output) {
    logger.setOwnerFuncName(__func__);
    if(img==nullptr || sizes<=0){
        logger.error("inputImage failed,img is null or size is 0");
        return;
    }
    const std::vector buf(img, img + sizes);  //构造临时缓冲区
    output = imdecode(buf, cv::IMREAD_COLOR);  // 修正参数为正确解码标志
    const int target_w = this->inputWidth;
    const int target_h = this->inputHeight;
    int src_w = output.cols;
    int src_h = output.rows;

    // 计算缩放因子（取宽高方向的最小比例，确保图像完全放入目标尺寸）
    double scale = std::min(static_cast<double>(target_w) / src_w,
                            static_cast<double>(target_h) / src_h);

    const int scaled_w = static_cast<int>(src_w * scale);
    const int scaled_h = static_cast<int>(src_h * scale);

    // 4. 计算填充量（上下左右填充黑边，使图像居中）
    const int top = (target_h - scaled_h) / 2;
    const int bottom = target_h - scaled_h - top;  // 处理奇数尺寸差异
    const int left = (target_w - scaled_w) / 2;
    const int right = target_w - scaled_w - left;

    Mat scaled;
    cv::resize(output, scaled, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(scaled, output, top, bottom, left, right,
                      cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void DnnYuNet::SetBlob(define::BasicDnnParams& params){
    logger.setOwnerFuncName(__func__);
     auto& paramsYuNet=dynamic_cast<IDnnParamsYuNet&>(params);
     this->detector->setInputSize(paramsYuNet.input.size());
}

void DnnYuNet::Process(define::BasicDnnParams& params) {
    logger.setOwnerFuncName(__func__);
    auto& paramsYuNet=dynamic_cast<IDnnParamsYuNet&>(params);
    this->detector->detect(paramsYuNet.input,paramsYuNet.faces);
    float scaleX = static_cast<float>(paramsYuNet.input.cols) / static_cast<float>(this->inputWidth);
    float scaleY = static_cast<float>(paramsYuNet.input.rows) / static_cast<float>(this->inputHeight);

    for (int i = 0; i < paramsYuNet.faces.rows; i++) {
        int x = static_cast<int>(paramsYuNet.faces.at<float>(i, 0) * scaleX);
        int y = static_cast<int>(paramsYuNet.faces.at<float>(i, 1) * scaleY);
        int w = static_cast<int>(paramsYuNet.faces.at<float>(i, 2) * scaleX);
        int h = static_cast<int>(paramsYuNet.faces.at<float>(i, 3) * scaleY);

        cv::rectangle(paramsYuNet.input, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
        paramsYuNet.facesRect.emplace_back(x, y, w, h);
        // 绘制关键点
        std::vector<cv::Point2f> landmarks;
        for (int j = 4; j < 14; j += 2) {
            float px = paramsYuNet.faces.at<float>(i, j) * scaleX;
            float py = paramsYuNet.faces.at<float>(i, j + 1) * scaleY;
            landmarks.emplace_back(px, py);

            cv::circle(paramsYuNet.input, cv::Point2f(px, py), 3, cv::Scalar(0, 0, 255), -1);
        }
        // 连接关键点 - 创建人脸轮廓
        // YuNet的关键点顺序通常是: 右眼、左眼、鼻子、右嘴角、左嘴角
        // 索引: 0-右眼, 1-左眼, 2-鼻子, 3-右嘴角, 4-左嘴角

        // 1. 连接双眼
        // cv::line(paramsYuNet.input, landmarks[0], landmarks[1], cv::Scalar(255, 0, 0), 2);
        //
        // // 2. 连接鼻子到双眼中心点（可选）
        // cv::Point2f eyesCenter = (landmarks[0] + landmarks[1]) * 0.5;
        // cv::line(paramsYuNet.input, landmarks[2], eyesCenter, cv::Scalar(255, 0, 0), 1);
        //
        // // 3. 连接嘴巴（左右嘴角）
        // cv::line(paramsYuNet.input, landmarks[3], landmarks[4], cv::Scalar(255, 0, 0), 2);
        //
        // // 4. 连接右眼到右嘴角
        // cv::line(paramsYuNet.input, landmarks[0], landmarks[3], cv::Scalar(255, 0, 0), 1);
        //
        // // 5. 连接左眼到左嘴角
        // cv::line(paramsYuNet.input, landmarks[1], landmarks[4], cv::Scalar(255, 0, 0), 1);
        //
        // // 6. 连接鼻子到嘴巴中心
        // cv::Point2f mouthCenter = (landmarks[3] + landmarks[4]) * 0.5;
        // cv::line(paramsYuNet.input, landmarks[2], mouthCenter, cv::Scalar(255, 0, 0), 1);

        // cv::line(paramsYuNet.input, landmarks[0], landmarks[1], cv::Scalar(255, 0, 0), 1);
        auto lines=[&](int i1,int i2){
            cv::line(paramsYuNet.input, landmarks[i1], landmarks[i2], cv::Scalar(255, 0, 0), 1);
        };
        lines(0,1); //双眼
        lines(0,2); //鼻子到双眼中心点
        lines(1,2);
        lines(3,2);
        lines(4,2);
        // lines(1,4);
        // lines(2,3);
        // lines(2,4);
    }
}

void DnnYuNet::PostProcessing(define::BasicDnnParams& params) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsYuNet&>(params);
    size_t totalBytes = 0;
    std::vector<size_t> featureSizes;
    for(auto& vec : dnnParams.faceFeatures) {
        size_t singleSize = vec.size() * sizeof(float);
        featureSizes.push_back(singleSize);
        totalBytes += singleSize;
    }
    dnnParams.faceFeatureByte.data = std::make_unique<unsigned char[]>(totalBytes);
    dnnParams.faceFeatureByte.dataSize =static_cast<int>(totalBytes);
    dnnParams.faceFeatureByte.faceNum=static_cast<int>(dnnParams.faceFeatures.size());
    // 分段拷贝特征数据
    size_t offset = 0;
    for(size_t i = 0; i < dnnParams.faceFeatures.size(); ++i) {
        const size_t copySize = featureSizes[i];
        memcpy(dnnParams.faceFeatureByte.data.get() + offset,
               dnnParams.faceFeatures[i].data(),
               copySize);
        offset += copySize;
    }
}
