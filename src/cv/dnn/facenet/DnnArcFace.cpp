//
// Created by HP on 2025/11/19.
//
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "smartsecurity/cv/dnn/face/dnnArcFace.h"
#include "smartsecurity/cv/dnn/face/dnnSSDcaffem.h"
#include "smartsecurity/cv/dnn/face/dnnYuNet.h"
using cv_dnn::DnnArcFace;
using cv_dnn::IDnnParamsSSDcaffem;
using cv::cvtColor;
using cv::dnn::blobFromImage;
using cv::Size;
using cv::Scalar;
DnnArcFace::DnnArcFace(const char* path) : DnnFaceNet(path) {
    logger.setOwnerName("DnnArcFace");
    logger.setOwnerFuncName(__func__);
}

void DnnArcFace::Process(define::BasicDnnParams& params)
{
    logger.setOwnerFuncName(__func__);
    logger.info("DnnArcFace::Process");
    auto& dnnParams=dynamic_cast<IDnnParamsSSDcaffem&>(params);
    auto faces=dnnParams.faces;
    for (auto faceRect : faces){
        Mat faceROI = dnnParams.input(faceRect).clone();
        Mat img_rgb;
        cvtColor(faceROI, img_rgb, cv::COLOR_BGR2RGB);
        Mat blob = blobFromImage(
            img_rgb,
            1.0,
            Size(inputWidth, inputHeight),
            false, false);
        net.setInput(blob);
        Mat featureVector = this->net.forward();
        vector featureVec(
            featureVector.ptr<float>(),
            featureVector.ptr<float>() + featureVector.total()
        );
        logger.info("featureVector size: "+to_string(featureVec.size()));
        dnnParams.faceFeatures.push_back(featureVec);
    }
}

void DnnArcFace::ProcessA(define::BasicDnnParams& params) {
    logger.setOwnerFuncName(__func__);
    auto& dnnParams=dynamic_cast<IDnnParamsYuNet&>(params);
    auto faces=dnnParams.facesRect;
    for (int i = 0; i < faces.size(); ++i){
        Mat faceROI = dnnParams.input(faces[i]).clone();
        auto faceRect = faces[i];
        vector<Point2f> currentLandmarks;
        // 注意：YuNet的faces矩阵中，每行是一个人脸，格式为[x, y, w, h, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, score]
        // 其中x0~y4对应5个关键点坐标（相对于原始输入图像）
        for (int j = 0; j < 5; ++j) {
            // YuNet的faces每行格式：[x, y, w, h, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, score]
            float x = dnnParams.faces.at<float>(i, 4 + 2 * j);  // 第i张脸的第j个关键点x
            float y = dnnParams.faces.at<float>(i, 5 + 2 * j);  // 第i张脸的第j个关键点y
            // 转换为ROI内的相对坐标（减去人脸框左上角）
            currentLandmarks.emplace_back(x - static_cast<float>(faceRect.x), y - static_cast<float>(faceRect.y));
        }
        // 3. 计算仿射变换矩阵（从当前关键点到标准关键点）
        // 注意：标准关键点的坐标是基于112x112的图像，需确保与对齐后的尺寸匹配
        Mat affineMatrix = cv::estimateAffinePartial2D(
            currentLandmarks,  // 输入：当前人脸关键点
            standardLandmarks  // 输出：标准关键点
        );
        if (affineMatrix.empty()) {
            logger.error("Affine matrix is empty!");
            continue;
        }else
        {
            logger.info("is affineMatrix empty: "+to_string(affineMatrix.empty()));
        }
        // 4. 执行仿射变换，得到对齐后的人脸（尺寸与标准关键点匹配，这里用112x112）
        Mat alignedFace;
        cv::warpAffine(
            faceROI,                // 输入：原始人脸ROI
            alignedFace,            // 输出：对齐后的人脸
            affineMatrix,           // 仿射矩阵
            Size(112, 112),     // 对齐后的尺寸（需与标准关键点匹配）
            cv::INTER_LINEAR,       // 插值方式
            cv::BORDER_CONSTANT,    // 边界填充方式
            Scalar(0, 0, 0)     // 边界填充颜色
        );

        Mat img_rgb;
        cvtColor(alignedFace, img_rgb, cv::COLOR_BGR2RGB);
        Mat blob = blobFromImage(
            img_rgb,
            1.0,
            Size(inputWidth, inputHeight),
            false, false);
        net.setInput(blob);
        Mat featureVector = this->net.forward();
        vector featureVec(
            featureVector.ptr<float>(),
            featureVector.ptr<float>() + featureVector.total()
        );
        logger.info("featureVector size: "+to_string(featureVec.size()));
        dnnParams.faceFeatures.push_back(featureVec);
    }
}

