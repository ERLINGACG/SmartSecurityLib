#ifndef DNN_DETECTION_H_
#define DNN_DETECTION_H_
#include <opencv2/dnn.hpp>
#include "smartsecurity/cv/cvexport.h"
#include "smartsecurity/cv/dnn/dnnBasicClass.h"
#include "nlohmann/json.hpp"
#include <mutex>
using namespace cv;
using namespace std;
using namespace dnn;
using namespace cv_dnn;
namespace dnnDetection{
    class DnnDetectorYolo :public dnnBasic::DnnBasicClass{

        double confThreshold{};
        double nmsThreshold{};
        std::mutex m_mutex; // 互斥锁
        std::vector<string> classNames;

        public:
            explicit DnnDetectorYolo(const char* jsonPath);
             ~DnnDetectorYolo() override =default;


            void InitModelYolo(const char* jsonPath);
            void InitConfig();

             void inputImage(unsigned char *inputData, int size, cv::Mat &orgImage);

             void outputImage(cv::Mat& inputImg,
                              std::vector<int>& class_ids,//分类类别索引
                              std::vector<float>& confidences,//置信度
                              std::vector<cv::Rect>& boxes,
                              std::vector<int>& nms_result,
                              data::ImageData& OutputData,
                              data::json::OutputJson& json
                              );//边框坐标信息



             void SetBlob(cv::Mat& blob,cv::Mat& inputImg);
             void ForwardYolo(std::vector<cv::Mat>& output_mat);
             void ProcessResults( cv::Mat& inputImg,
                                  std::vector<Mat>& output_mat,
                                  std::vector<int>& class_ids,//分类类别索引
                                  std::vector<float>& confidences,//置信度
                                  std::vector<cv::Rect>& boxes,//边框坐标信息
                                  std::vector<int>& nms_result);


             void DetectImage_3(
                     unsigned char* inputData,int size,
                     data::ImageData& OutputData,
                     data::json::OutputJson& json
                     );
             void DetectImage(
                     unsigned char* inputData,int size,
                     data::ImageData& OutputData) {};
             void DetectImage(
                    unsigned char* inputData,int size,
                    data::ImageData& OutputData,
                    data::json::OutputJson& json);


    };

}


#endif