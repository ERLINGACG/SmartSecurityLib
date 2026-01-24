//
// Created by HP on 2025/11/24.
//

#ifndef DNNSR_H
#define DNNSR_H
#include <nlohmann/json.hpp>
#include "opencv2/dnn_superres.hpp"
using cv::dnn_superres::DnnSuperResImpl;
using nlohmann::json;
using std::ifstream;
using Ilogger::DynamicLogger;
namespace cv_sr
{
    class DnnSR
    {
        json config;
        DynamicLogger logger;
        DnnSuperResImpl sr;
        public:
            explicit DnnSR(const char* path);
            ~DnnSR()=default;


            void InitConfig();
            void Load(const char* path);
            void Upsample(cv::Mat& src, cv::Mat& dst);

    };
}
#endif //DNNSR_H
