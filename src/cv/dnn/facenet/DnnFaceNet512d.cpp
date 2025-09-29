//
// Created by HP on 2025/11/17.
//
#include "opencv2/imgproc.hpp"
#include "../../../../include/smartsecurity/cv/dnn/face/dnnFaceNet512d.h"
#include "../../../../include/smartsecurity/cv/dnn/face/dnnSSDcaffem.h"
#include "smartsecurity/cv/dnn/face/dnnYuNet.h"
using cv_dnn::define::BasicDnnParams;
using cv_dnn::DnnFaceNet512d;
using cv_dnn::DnnFaceNet;
using cv::Mat;
using cv::Size;
using cv::Scalar;
using cv::dnn::blobFromImage;
using cv::cvtColor;
using cv::COLOR_BGR2RGB;
DnnFaceNet512d::DnnFaceNet512d(const char *path) : DnnFaceNet(path)  // 显式初始化父类
{
    // 这里可以添加子类特有的初始化逻辑（父类已完成初始化）
    logger.setOwnerName("DnnFaceNet512d");
    logger.setOwnerFuncName(__func__);
}

// 原日志错误，
// [ERROR:0@0.672] global net_impl.cpp:1165 cv::dnn::dnn4_v20241223::Net::Impl::getLayerShapesRecursively OPENCV/DNN: [Convolution]:(onnx_node!StatefulPartitionedCall/inception_resnet_v1/Conv2d_1a_3x3/Conv2D): getMemoryShapes() throws exception. inputs=1 outputs=0/1 blobs=2
// [ERROR:0@0.673] global net_impl.cpp:1168 cv::dnn::dnn4_v20241223::Net::Impl::getLayerShapesRecursively     input[0] = [ 1 160 3 160 ]
// [ERROR:0@0.673] global net_impl.cpp:1176 cv::dnn::dnn4_v20241223::Net::Impl::getLayerShapesRecursively     blobs[0] = CV_32FC1 [ 32 3 3 3 ]
// [ERROR:0@0.673] global net_impl.cpp:1176 cv::dnn::dnn4_v20241223::Net::Impl::getLayerShapesRecursively     blobs[1] = CV_32FC1 [ 32 1 ]
// [ERROR:0@0.673] global net_impl.cpp:1178 cv::dnn::dnn4_v20241223::Net::Impl::getLayerShapesRecursively Exception message: OpenCV(4.12.0-dev) E:\CPPlib\SelfLib\opencv\opencv-4.x\modules\dnn\src\layers\convolution_layer.cpp:396: error: (-2:Unspecified error) Number of input channels should be multiple of 3 but got 160 in function 'cv::dnn::ConvolutionLayerImpl::getMemoryShapes'

// 适配方案
void DnnFaceNet512d::Process(BasicDnnParams& params)
{
    logger.setOwnerFuncName("DnnFaceNet512d::Process");
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
    auto faces=dnnParams.faces;
    for (auto faceRect : faces)
    {
      cv::Mat faceROI = dnnParams.input(faceRect).clone();

    // 2. BGR转RGB
        Mat img_rgb;
        cvtColor(faceROI, img_rgb, cv::COLOR_BGR2RGB);

        // 3. 调整尺寸为160x160（双线性插值）
        Mat img_resized;
        cv::resize(img_rgb, img_resized, Size(this->inputWidth, this->inputHeight), 0, 0, cv::INTER_LINEAR);

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
        // 4. 设置输入并推理（此时blob是NHWC格式，与OpenCV DNN兼容）
        Mat input_blob = Mat(1, 160*160*3, CV_32F, input_data.data())
                     .reshape(1, {1, 160, 160, 3});
        this->net.setInput(input_blob);
        Mat featureVector = this->net.forward();
        vector featureVec(
            featureVector.ptr<float>(),
            featureVector.ptr<float>() + featureVector.total()
        );
        string featureStr;
        for (auto f: featureVec)
        {
            featureStr+=to_string(f)+",";
        }
        logger.info("featureVector: "+featureStr);
        logger.info("featureVector size: "+to_string(featureVec.size()));
        dnnParams.faceFeatures.push_back(featureVec);
    }

}

