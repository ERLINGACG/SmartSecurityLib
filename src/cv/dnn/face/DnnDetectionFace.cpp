//
// Created by HP on 2025/8/15.
//
#include <iostream>
#include <fstream>
#include "smartsecurity/cv/dnn/dnnDetectionFace.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace  cv_dnn::face;

DnnDetectorFace::DnnDetectorFace(cv_dnn::param::face::FaceParam* param){
    this->InitModelFace(param);
}


void DnnDetectorFace::DetectImage(unsigned char *inputData, int size, data::ImageData &OutputData) {

}

void DnnDetectorFace::InitModelFace(cv_dnn::param::face::FaceParam *pParam) {
    this->LoadModelFace(pParam);
}
void DnnDetectorFace::LoadModelFace(cv_dnn::param::face::FaceParam *param) {
    std::cout << "caffemodel_path:" << param->caffemodel_path << std::endl;
    std::cout << "prototxt_path:" << param->prototxt_path << std::endl;
    std::cout<<"isCuda:"<<param->isCuda<<std::endl;
    try{
        this->net=cv::dnn::readNetFromCaffe(param->prototxt_path,param->caffemodel_path);
        if (this->net.empty()) {
            std::cerr << "Failed to load model!" << std::endl;
        }else{
            std::cout<<"net load success"<<std::endl;
        }
        if(param->isCuda){
            this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout<<"dnn backend cuda"<<std::endl;
        }
    }catch (const cv::Exception& e){
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return;
    }

}
void DnnDetectorFace::Load(nlohmann::json &config) {
    try{
        nlohmann::json activity=config["env"];
        nlohmann::json env=config[activity];
        std::cout<<"env:"<<env<<std::endl;
        std::cout<<"prototxt_path:"<<env["prototxt_path"]<<std::endl;
        std::cout<<"caffemodel_path:"<<env["caffemodel_path"]<<std::endl;
        std::cout<<"isCuda:"<<env["isCuda"]<<std::endl;
        this->net=cv::dnn::readNetFromCaffe(std::string(env["prototxt_path"]),
                                            std::string (env["caffemodel_path"]));
        if(env["isCuda"]){
            this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout<<"dnn backend cuda"<<std::endl;
        }
    }catch(const std::exception& e){
        std::cout<<"load config file failed"<<e.what()<<std::endl;
    }
}

void DnnDetectorFace::getFaceFeature(unsigned char *inputData, int size, cv_param::EncodeParam *encodeParam,
                                      cv_dnn::face::DnnFeatureFace *faceFeature, data::ImageData &OutputData,
                                      data::FaceFeatureByte &faceFeatureByte) {
    if (inputData == nullptr) {
        std::cout << "inputData is nullptr" << std::endl;
        return;
    }
    cv::Mat image;
    cv::Mat blob;
    cv::Mat output;
    std::vector<cv::Rect> faces;
    cv_dnn::face::DnnDetectorFace::inputImage(inputData,size,image);
    if (image.empty()) {
        std::cerr << "Input image is empty after decoding" << std::endl;
        return;
    }
    cv_dnn::face::DnnDetectorFace::resizeWithPadding(image,300,300);
    this->SetBlob(blob,image);
    this->ForwardFace(output);
    DnnDetectorFace::ProcessResults(output,image,0.6,faces);
    ProcessResults_getFeature(faceFeature,faces,image,faceFeatureByte);
    outputImage(OutputData,image,encodeParam);
}

void DnnDetectorFace::inputImage(unsigned char *inputData, int size, cv::Mat &orgImage) {
    try{
        orgImage=cv::imdecode(cv::Mat(1,size,CV_8UC1,inputData),cv::IMREAD_COLOR);
        // 新增检查：如果图像为空，给出明确错误信息
        if (orgImage.empty()) {
            std::cerr << "Failed to decode image. Input data size: " << size << std::endl;
        }
    }catch (std::exception& e){
        std::cerr << "OpenCV error: for inputImage " << e.what() << std::endl;
    }
}

void DnnDetectorFace::resizeWithPadding(cv::Mat &orgImage,int inputWidth,int inputHeight) {
    try{
        if (orgImage.empty()) {
            std::cerr << "Cannot resize empty image" << std::endl;
            return;
        }
        double  scale =std::min(static_cast<double>(inputWidth)/ orgImage.cols,
                                static_cast<double>(inputHeight)/ orgImage.rows);

        cv::Size scaledSize(static_cast<int>(orgImage.cols*scale),
                            static_cast<int>(orgImage.rows*scale));
        //确定缩放因子
        cv::resize(orgImage,orgImage,scaledSize,0, 0, cv::INTER_LINEAR); //缩放图像

        int padH = inputHeight - orgImage.rows;
        int padW = inputWidth - orgImage.cols;
        // 填充图像
        cv::copyMakeBorder(orgImage, orgImage,
                           padH/2, padH - padH/2,  // 垂直方向（上下）
                           padW/2, padW - padW/2,  // 水平方向（左右）
                           cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    }catch (std::exception& e){
        std::cerr << "OpenCV error: for resizeWithPadding " << e.what() << std::endl;
    }
}

void DnnDetectorFace::outputImage(data::ImageData &OutputData,cv::Mat &orgImage, cv_param::EncodeParam *encodeParam){
    try{
        if (orgImage.empty()) {
            std::cerr << "Cannot encode empty image" << std::endl;
            return;
        }
        std::vector<int> params_ {cv::IMWRITE_WEBP_QUALITY, encodeParam->qos};
        std::vector<unsigned char> encoded;
        cv::imencode(encodeParam->type, orgImage, encoded, params_);
        OutputData.size =  static_cast<int>(encoded.size());
        OutputData.width = orgImage.cols;
        OutputData.height = orgImage.rows;
        OutputData.channels = 3;
        OutputData.data = std::make_unique<unsigned char[]>(encoded.size());
        memcpy(OutputData.data.get(), encoded.data(), encoded.size());
    }catch (std::exception& e){
        std::cerr << "OpenCV error: for outputImage " << e.what() << std::endl;
    }
}

void DnnDetectorFace::SetBlob(cv::Mat& blob,cv::Mat& orgImage) {
    try {
        blob = cv::dnn::blobFromImage(orgImage, 1.0,
                                      cv::Size(300, 300),
                                      cv::Scalar(104, 117, 123), false, false);
        this->net.setInput(blob);
    } catch (std::exception &e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    }
}

void DnnDetectorFace::ForwardFace(cv::Mat& output)  {
    try{
        output=this->net.forward();
    }catch (std::exception& e){
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    }
}


void DnnDetectorFace::ProcessResults(cv::Mat &output, cv::Mat& orgImage,double confidence_threshold,std::vector<cv::Rect>& _faces){
    cv::Mat detections(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    if(detections.empty()){
        std::cout << "No faces detected" << std::endl;
        return;
    }
    std::vector<cv::Rect> faces;

    std::vector<float> confidences;
    for(int i=0;i<detections.rows;i++){
        float confidence=detections.at<float>(i,2);
        if(confidence > confidence_threshold) {
            int x1 = static_cast<int>(detections.at<float>(i, 3) *  static_cast<float>(orgImage.cols));
            int y1 = static_cast<int>(detections.at<float>(i, 4) *  static_cast<float>(orgImage.rows));
            int x2 = static_cast<int>(detections.at<float>(i, 5) *  static_cast<float>(orgImage.cols));
            int y2 = static_cast<int>(detections.at<float>(i, 6) *  static_cast<float>(orgImage.rows));

            faces.emplace_back(x1, y1, x2 - x1, y2 - y1);
            confidences.emplace_back(confidence);
            // 绘制检测结果（可选）
            cv::rectangle(orgImage, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        }
    }
    _faces=faces; //应该移到这里处理，否则_faces为空
}

void DnnDetectorFace::ProcessResults_getFeature(
        cv_dnn::face::DnnFeatureFace* featureExtractor,
        std::vector<cv::Rect>& faces,
        cv::Mat& orgImage,data::FaceFeatureByte& faceFeatureByte
        ) {
    std::vector<std::vector<float>> faceFeatures;
    if(faces.empty()){
        std::cout << "No faces detected, faceFeatures size: " << faceFeatures.size() << std::endl;
        faceFeatureByte.dataSize=0;
        faceFeatureByte.faceNum=0;
        faceFeatureByte.data= std::make_unique<unsigned char[]>(0);
        return;
    }
    int faceNum=0;
    for(size_t i = 0; i < faces.size(); ++i) {
        const auto &faceRect = faces[i];
        // 裁剪人脸区域
        cv::Mat faceROI = orgImage(faceRect).clone();
        // 预处理人脸图像（根据特征提取模型的要求）
        cv::Mat inputBlob = cv::dnn::blobFromImage(
                faceROI,
                1.0 / 128,              // 缩放因子
                cv::Size(160, 160),      // 特征提取模型输入尺寸
                cv::Scalar(127.5, 127.5, 127.5), // 均值
                true,                    // 交换R和B通道
                false                    // 不裁剪
        );
        // 设置输入并前向传播
        featureExtractor->getNet().setInput(inputBlob);
        cv::Mat featureVector = featureExtractor->getNet().forward();

        // 将特征向量转换为std::vector<float>
        std::vector<float> featureVec(
                featureVector.ptr<float>(),
                featureVector.ptr<float>() + featureVector.total()
        );

        faceFeatures.push_back(featureVec);
        std::cout << "Extracted feature vector for face " << i
                  << " (size: " << featureVec.size() << ")" << std::endl;
        faceNum++;
    }
    size_t totalBytes = 0;
    std::vector<size_t> featureSizes;
    for(auto& vec : faceFeatures) {
        size_t singleSize = vec.size() * sizeof(float);
        featureSizes.push_back(singleSize);
        totalBytes += singleSize;
    }
    // 创建连续内存空间
    faceFeatureByte.data = std::make_unique<unsigned char[]>(totalBytes);
    faceFeatureByte.dataSize =static_cast<int>( totalBytes);
    faceFeatureByte.faceNum=faceNum;
    // 分段拷贝特征数据
    size_t offset = 0;
    for(size_t i = 0; i < faceFeatures.size(); ++i) {
        size_t copySize = featureSizes[i];
        memcpy(faceFeatureByte.data.get() + offset,
               faceFeatures[i].data(),
               copySize);
        offset += copySize;
    }
}
double DnnDetectorFace::getDistanceForByte(unsigned char* input_1,
                                           unsigned char* input_2) {

    auto* feat1 = reinterpret_cast<float*>(input_1);
    auto* feat2 = reinterpret_cast<float*>(input_2);

    cv::Mat vec1(1, 128, CV_32F, feat1);
    cv::Mat vec2(1, 128, CV_32F, feat2);
    double euclidean_dist = cv::norm(vec1 - vec2);
    return euclidean_dist; // 或返回 1 - euclidean_dist
}

void DnnDetectorFace::LoadJson(const char *path) {
    std::cout<<"load json file "<<path<<std::endl;
    std::unique_ptr<std::ifstream> jsonfile(new std::ifstream(path));
    if(jsonfile->is_open()){
        nlohmann::json j;
        *jsonfile >> j;
        try{
            Load(j);
        }catch(const std::exception& e){
            std::cout<<"load config file failed"<<e.what()<<std::endl;
        }

    }else{
        std::cout<<"no config file "<<path<<std::endl;
    }
}

void DnnDetectorFace::InitModelFace(const char *path) {
    this->LoadJson(path);
}

DnnDetectorFace::DnnDetectorFace(const char *path) {
    this->InitModelFace(path);
}


