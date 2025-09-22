//
// Created by HP on 2025/8/15.
//

#ifndef SMARTSECURITYCORELIB_DNN_DETECTION_FACE_H
#define SMARTSECURITYCORELIB_DNN_DETECTION_FACE_H
#include "smartsecurity/cv/dnn/dnnBasicClass.h"
#include "smartsecurity/cv/param/dnnParam.h"
#include "smartsecurity/cv/param/encodeParam.h"
#include "dnnFeatureFace.h"

namespace cv_dnn::face{
class DnnDetectorFace : public cv_dnn::dnnBasic::DnnBasicClass{
    public:
         explicit DnnDetectorFace(cv_dnn::param::face::FaceParam* param); //重写构造函数
         explicit DnnDetectorFace(const char* path);
        ~DnnDetectorFace() override =default;
        void DetectImage(unsigned char* inputData, int size, data::ImageData& OutputData) override;
        void InitModelFace(param::face::FaceParam *pParam);
        void InitModelFace(const char* path);
        void LoadModelFace(param::face::FaceParam *param);



        void LoadJson(const char *path) override;
        void Load(nlohmann::json &config) override;
        void getFaceFeature(unsigned char* inputData, int size,
                             cv_param::EncodeParam* encodeParam,
                             cv_dnn::face::DnnFeatureFace* faceFeature,
                             data::ImageData& OutputData,data::FaceFeatureByte  & faceFeatureByte);

        void inputImage(unsigned char *inputData, int size, cv::Mat &orgImage) override;

        static void resizeWithPadding(cv::Mat &orgImage,int inputWidth,int inputHeight);



        void SetBlob(cv::Mat& blob,cv::Mat& orgImage) override;

        void Forward(cv::Mat& output) override;

        static void outputImage(data::ImageData &OutputData,cv::Mat &orgImage, cv_param::EncodeParam *encodeParam);
        void ProcessResults(cv::Mat &output, cv::Mat& orgImage,double confidence_threshold,std::vector<cv::Rect>& _faces) override;

        static void ProcessResults_getFeature(
            cv_dnn::face::DnnFeatureFace* featureExtractor,
            std::vector<cv::Rect>& faces,
            cv::Mat& orgImage,data::FaceFeatureByte& faceFeatureByte);

        static double getDistanceForByte(unsigned char *input_1, unsigned char *input_2);
    };
}

#endif //SMARTSECURITYCORELIB_DNN_DETECTION_FACE_H
