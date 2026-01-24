//
// Created by HP on 2025/11/14.
//
#include "opencv2/imgproc.hpp"
#include "../../../../include/smartsecurity/cv/dnn/face/dnnFaceNet.h"
#include "../../../../include/smartsecurity/cv/dnn/face/dnnSSDcaffem.h"
using cv_dnn::IDnnParamsSSDcaffem;
using cv::dnn::readNetFromONNX;
using cv::dnn::blobFromImage;
using cv::Size;
using cv::Scalar;
cv_dnn::DnnFaceNet::DnnFaceNet(const char* path) {
    logger.setOwnerName("DnnFaceNet");
    logger.setOwnerFuncName(__func__);
    this->Load(path);
    this->DnnFaceNet::InitConfig();
}

void cv_dnn::DnnFaceNet::InitConfig(){
    string activity=config["env"];
    logger.info("use activity "+activity);
    json env=config[activity];
    if (env.empty())
    {
        logger.error("env is empty");
        return;
    }
    this->net=readNetFromONNX(string(env["facenet_path"]));
    this->inputWidth=env["input_width"];
    this->inputHeight=env["input_height"];
    if(this->net.empty()){
        logger.error("readNetFromONNX failed");
        return;
    }
    if (env["isCUDA"]){
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        logger.info("dnn backend cuda");
    }else{
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        logger.info("dnn backend cpu");
    }
    logger.info("facenet init config success");
}

void cv_dnn::DnnFaceNet::PostProcessing(define::BasicDnnParams& params){
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
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

void cv_dnn::DnnFaceNet::UnPacketFeature(define::BasicDnnParams& params,
                                         FaceFeatureByte& faceFeatureByte){
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
    faceFeatureByte=std::move(dnnParams.faceFeatureByte);
}

void cv_dnn::DnnFaceNet::Process(define::BasicDnnParams& params) {
        logger.setOwnerFuncName(__func__);
        auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
        auto  faces=dnnParams.faces;
        for (auto faceRect : faces) {
            Mat faceROI = dnnParams.input(faceRect).clone();
            // Mat blob = blobFromImage(faceROI,
            //     1.0,
            //     Size(this->inputWidth, this->inputHeight),
            //     Scalar(104.0, 177.0, 123.0),
            //     false, false
            // );
            // 1. 缩放图像到模型输入尺寸 (W, H)
            Mat img_rgb;
            cvtColor(faceROI, img_rgb, cv::COLOR_BGR2RGB);

            // 3. 调整尺寸为160x160（双线性插值）
            Mat img_resized;
            cv::resize(img_rgb, img_resized, Size(this->inputWidth, this->inputHeight), 0, 0, cv::INTER_LINEAR);
            img_resized -= Scalar(104.0, 177.0, 123.0); // 归一化到[-128, 127]范围
            // 4. 归一化到[-1, 1]范围
            cv::Mat img_normalized;
            img_resized.convertTo(img_normalized, CV_32F, 1.0 / 255.0);  // 先转换为[0,1]
            img_normalized = img_normalized * 2.0 - 1.0;                // 再转换为[-1,1]

            // 5. 按NHWC格式展平为一维数组（H->W->C顺序，C为RGB）
            std::vector<float> input_data;
            input_data.reserve(this->inputHeight * this->inputWidth * 3);  // 预分配内存
            for (int h = 0; h < this->inputHeight; ++h) {
                for (int w = 0; w < this->inputWidth; ++w) {
                    // 获取当前像素的RGB值（注意OpenCV的Mat存储顺序是行优先）
                    cv::Vec3f pixel = img_normalized.at<cv::Vec3f>(h, w);
                    input_data.push_back(pixel[0]);  // R通道
                    input_data.push_back(pixel[1]);  // G通道
                    input_data.push_back(pixel[2]);  // B通道
                }
            }
            // 4. 设置输入并推理（此时blob是NCHW格式，与OpenCV DNN兼容）
            Mat input_blob = Mat(1, 160*160*3, CV_32F, input_data.data())
                         .reshape(1, {1, 3, 160, 160});
            this->net.setInput(input_blob);
            Mat featureVector = this->net.forward();
            vector featureVec(
                featureVector.ptr<float>(),
                featureVector.ptr<float>() + featureVector.total()
            );
            dnnParams.faceFeatures.push_back(featureVec);
        }
}

