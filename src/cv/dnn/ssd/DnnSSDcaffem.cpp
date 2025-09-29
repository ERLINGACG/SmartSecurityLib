//
// Created by HP on 2025/11/14.
//
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "../../../../include/smartsecurity/cv/dnn/face/dnnFaceNet512d.h"
#include "../../../../include/smartsecurity/cv/dnn/face/dnnSSDcaffem.h"
#include "smartsecurity/cv/dnn/face/dnnArcFace.h"
using cv::imdecode;
using cv::dnn::blobFromImage;
using cv::imdecode;
cv_dnn::DnnSSDcaffem::DnnSSDcaffem(const char* path,DnnFaceNet* faceNet) {
    this->logger.setOwnerName("DnnSSDcaffem");
    this->Load(path);
    this->DnnSSDcaffem::InitConfig();
    this->faceNet=faceNet;
}

void cv_dnn::DnnSSDcaffem::DebugTime(int size, unsigned char* img, ImageData& OutputData, OutputJson& json){
      logger.setOwnerName(__func__);
      auto test=new IDnnParamsSSDcaffem();
      logger.time(&DnnSSDcaffem::Input,this,size,img,test->input);
      logger.time(&DnnSSDcaffem::PacketData,this,*test,test->input);
      logger.time(&DnnSSDcaffem::SetBlob,this,*test);
      logger.time(&DnnSSDcaffem::Forward,this,*test);
      logger.time(&DnnSSDcaffem::Process,this,*test);
      logger.time(&DnnSSDcaffem::PostProcessing,this,*test);
      logger.time(&DnnSSDcaffem::UnPacketData,this,*test,OutputData,json);
}


void cv_dnn::DnnSSDcaffem::InitConfig() {
    logger.setOwnerFuncName(__func__);
    string activity=config["env"];
    json env=config[activity];
    this->inputWidth=env["inputWidth"];
    this->inputHeight=env["inputHeight"];
    this->confThreshold =env["confThreshold"];
    this->net=cv::dnn::readNetFromCaffe(string(env["prototxt_path"]),string (env["caffemodel_path"]));

    if(env["isCUDA"]){
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        logger.info("dnn backend cuda");
    }else {
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        logger.info("dnn backend cpu");
    }

    if (!net.empty()) {
        logger.info("dnn init success");
    }else{
        logger.info("dnn init failed");
    }
}

void cv_dnn::DnnSSDcaffem::PacketData(define::BasicDnnParams& params, Mat& input) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
    dnnParams.input=input;
}

void cv_dnn::DnnSSDcaffem::UnPacketData(define::BasicDnnParams& params, ImageData& imageData, OutputJson& outputJson) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
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

void cv_dnn::DnnSSDcaffem::Input(int sizes, unsigned char* img, Mat& output) {
    logger.setOwnerFuncName(__func__);
    if(img==nullptr || sizes<=0){
        logger.error("inputImage failed,img is null or size is 0");
        return;
    }
    const std::vector buf(img, img + sizes);  //构造临时缓冲区
    output = imdecode(buf, cv::IMREAD_COLOR);  // 修正参数为正确解码标志
    // const int target_w = this->inputWidth;
    // const int target_h = this->inputHeight;
    // int src_w = output.cols;
    // int src_h = output.rows;
    //
    // // 计算缩放因子（取宽高方向的最小比例，确保图像完全放入目标尺寸）
    // double scale = std::min(static_cast<double>(target_w) / src_w,
    //                         static_cast<double>(target_h) / src_h);
    //
    // const int scaled_w = static_cast<int>(src_w * scale);
    // const int scaled_h = static_cast<int>(src_h * scale);
    //
    // // 4. 计算填充量（上下左右填充黑边，使图像居中）
    // const int top = (target_h - scaled_h) / 2;
    // const int bottom = target_h - scaled_h - top;  // 处理奇数尺寸差异
    // const int left = (target_w - scaled_w) / 2;
    // const int right = target_w - scaled_w - left;
    //
    // Mat scaled;
    // cv::resize(output, scaled, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_AREA);
    // cv::copyMakeBorder(scaled, output, top, bottom, left, right,
    //                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

}

void cv_dnn::DnnSSDcaffem::SetBlob(define::BasicDnnParams& params) {
   logger.setOwnerFuncName(__func__);
   auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
    dnnParams.blob  =blobFromImage(
       dnnParams.input,
       1.0, cv::Size(300, 300),
       cv::Scalar(104, 117, 123), false, false
    );
    this->net.setInput(dnnParams.blob);
}

void cv_dnn::DnnSSDcaffem::Forward(define::BasicDnnParams& params) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
    dnnParams.forward = this->net.forward();
}

void cv_dnn::DnnSSDcaffem::Process(define::BasicDnnParams& params) {
   logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
    Mat detections( dnnParams.forward.size[2],
                     dnnParams.forward.size[3], CV_32F,dnnParams.forward.ptr<float>());
    if(detections.empty()){
        logger.warn("No faces detected");
        return;
    }

    for(int i=0;i<detections.rows;i++){
        float confidence=detections.at<float>(i,2);
        if (confidence>this->confThreshold){
            int x1 = static_cast<int>(detections.at<float>(i, 3) * static_cast<float>(dnnParams.input.cols));
            int y1 = static_cast<int>(detections.at<float>(i, 4) * static_cast<float>(dnnParams.input.rows));
            int x2 = static_cast<int>(detections.at<float>(i, 5) * static_cast<float>(dnnParams.input.cols));
            int y2 = static_cast<int>(detections.at<float>(i, 6) * static_cast<float>(dnnParams.input.rows));
            dnnParams.faces.emplace_back(x1, y1, x2 - x1, y2 - y1);
            dnnParams.confidences.push_back(confidence);
            dnnParams.count++;
            dnnParams.message[dnnParams.count-1]=confidence;
            cv::rectangle(dnnParams.input, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            putText(dnnParams.input, "face"+std::to_string(dnnParams.count-1)+":"+std::to_string(confidence), cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
    }
}

void cv_dnn::DnnSSDcaffem::PostProcessing(define::BasicDnnParams& params) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
    if(dnnParams.count<=0){
        logger.warn("No faces detected");
        return;
    }
    json results;
    results["count"]=dnnParams.count;
    for(int i=0;i<dnnParams.count;i++){
        results["face:"+std::to_string(i)]=dnnParams.message[i];
    }
    auto buffer = std::make_unique<char[]>(results.dump().size() + 1);
    std::ranges::copy(results.dump(), buffer.get());
    buffer[results.dump().size()] = '\0';
    dnnParams.outputJson.json=std::move(buffer);
    dnnParams.outputJson.size=static_cast<int>(results.dump().size());
}

void cv_dnn::DnnSSDcaffem::Detection(int size, unsigned char* img, ImageData& OutputData, OutputJson& json){

}

void cv_dnn::DnnSSDcaffem::DetectionFeature(int size, unsigned char* img, ImageData& OutputData, OutputJson& json,
    FaceFeatureByte& faceFeatureByte) {
    logger.setOwnerFuncName(__func__);
    auto test=new IDnnParamsSSDcaffem();
    logger.time(&DnnSSDcaffem::Input,this,size,img,test->input);
    logger.time(&DnnSSDcaffem::PacketData,this,*test,test->input);
    logger.time(&DnnSSDcaffem::SetBlob,this,*test);
    logger.time(&DnnSSDcaffem::Forward,this,*test);
    logger.time(&DnnSSDcaffem::Process,this,*test);
    logger.time(&DnnSSDcaffem::PostProcessing,this,*test);

    logger.time([&] {
          this->faceNet->Process(*test);

    });// 不需要额外参数，因为 lambda 已经捕获了所需的一切
    logger.time([&] {
          this->faceNet->PostProcessing(*test);
    });

    logger.time(&DnnSSDcaffem::UnPacketData,this,*test,OutputData,json);
    logger.time([&] {
          this->faceNet->UnPacketFeature(*test,faceFeatureByte);
    });
}
void cv_dnn::DnnSSDcaffem::DetectionFeature512d(int size, unsigned char* img, ImageData& OutputData, OutputJson& json,
    FaceFeatureByte& faceFeatureByte) {

    logger.setOwnerFuncName(__func__);
    logger.info("DetectionFeature512d");
    auto test=new IDnnParamsSSDcaffem();
    logger.time(&DnnSSDcaffem::Input,this,size,img,test->input);
    logger.time(&DnnSSDcaffem::PacketData,this,*test,test->input);
    logger.time(&DnnSSDcaffem::SetBlob,this,*test);
    logger.time(&DnnSSDcaffem::Forward,this,*test);
    logger.time(&DnnSSDcaffem::Process,this,*test);
    logger.time(&DnnSSDcaffem::PostProcessing,this,*test);

    logger.time([&] {
          dynamic_cast<DnnFaceNet512d*>(this->faceNet)->Process(*test);

    });// 不需要额外参数，因为 lambda 已经捕获了所需的一切
    logger.time([&] {
          dynamic_cast<DnnFaceNet512d*>(this->faceNet)->PostProcessing(*test);
    });

    logger.time(&DnnSSDcaffem::UnPacketData,this,*test,OutputData,json);
    logger.time([&] {
          dynamic_cast<DnnFaceNet512d*>(this->faceNet)->UnPacketFeature(*test,faceFeatureByte);
    });
}

void cv_dnn::DnnSSDcaffem::DetectionFeatureArc(int size, unsigned char* img, ImageData& OutputData, OutputJson& json,
    FaceFeatureByte& faceFeatureByte)
{
    logger.setOwnerFuncName(__func__);
    logger.info("DetectionFeatureArc");
    auto test=new IDnnParamsSSDcaffem();
    logger.time(&DnnSSDcaffem::Input,this,size,img,test->input);
    logger.time(&DnnSSDcaffem::PacketData,this,*test,test->input);
    logger.time(&DnnSSDcaffem::SetBlob,this,*test);
    logger.time(&DnnSSDcaffem::Forward,this,*test);
    logger.time(&DnnSSDcaffem::Process,this,*test);
    logger.time(&DnnSSDcaffem::PostProcessing,this,*test);

    logger.time([&]
    {
        dynamic_cast<DnnArcFace*>(this->faceNet)->Process(*test);
    });
    logger.time([&]
    {
        dynamic_cast<DnnArcFace*>(this->faceNet)->PostProcessing(*test);
    });

    logger.time([&] {
          dynamic_cast<DnnArcFace*>(this->faceNet)->UnPacketFeature(*test,faceFeatureByte);
    });
}
