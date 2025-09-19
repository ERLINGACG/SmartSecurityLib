#ifndef DNN_DETECTION_H_
#define DNN_DETECTION_H_
#include <opencv2/dnn.hpp>
#include "opencv2/mcc/checker_model.hpp"
#include "smartsecurity/cv/cvexport.h"
#include "smartsecurity/cv/param/dnnParam.h"
#include "smartsecurity/cv/dnn/dnnBasicClass.h"
#include "nlohmann/json.hpp"
#include <mutex>
using namespace cv;
using namespace std;
using namespace dnn;
using namespace cv_dnn;
namespace dnnDetection{
    class DnnDetectorYolo : public cv_dnn::dnnBasic::DnnBasicClass{

        private:

             double confThreshold{};
             double nmsThreshold{};
             std::mutex m_mutex; // 互斥锁
//             std::vector<string> classNames = {
//                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//                "hair drier", "toothbrush"
//            };
              std::vector<string> classNames;
//            std::vector<string> classNames={"Provence", "Shu", "Logos", "Mon3tr", "Ulpianus", "Texas"};
        public:
            DnnDetectorYolo(const char* Yolo_path, bool isCUDA, double confThreshold, double nmsThreshold)  ;
            DnnDetectorYolo(const char* jsonPath);
             ~DnnDetectorYolo() override =default;
             void Load(nlohmann::json& j);
             static  void inputImage(unsigned char *inputData, int size, cv::Mat &orgImage);
             void outputImage(cv::Mat& inputImg,
                              std::vector<int>& class_ids,//分类类别索引
                              std::vector<float>& confidences,//置信度
                              std::vector<cv::Rect>& boxes,
                              std::vector<int>& nms_result,
                              data::ImageData& OutputData,
                              data::json::OutputJson& json);//边框坐标信息
             void LoadJson(const char* path) override{};
             void SetBlob(cv::Mat& blob,cv::Mat& inputImg) override;
             void Forward(std::vector<Mat>& output_mat);
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
                     data::ImageData& OutputData) override {};
             void DetectImage(
                    unsigned char* inputData,int size,
                    data::ImageData& OutputData,
                    data::json::OutputJson& json);


    };

}


#endif