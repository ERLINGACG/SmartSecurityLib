//
// Created by HP on 2025/11/24.
//
#include "log/dynamiclogger.h"
#include "smartsecurity/cv/dnn/sr/dnnsr.h"
using std::ifstream;
using Ilogger::DynamicLogger;
using cv_sr::DnnSR;

DnnSR::DnnSR(const char* path)
{
    this->logger.setOwnerName("DnnSR");
    this->Load(path);
    this->InitConfig();
}
void DnnSR::InitConfig()
{
    logger.setOwnerFuncName(__func__);
    string env   =config["env"];
    json _config =config[env];
    sr.readModel(_config["model_path"]);
    sr.setModel(_config["model_type"], _config["model_scale"]);
    if (_config["isCUDA"])
    {
        sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        logger.info("DnnSR::InitConfig Use CUDA");
    }
    logger.info("DnnSR::InitConfig Success,USE: "+_config["model_path"].get<std::string>());

}
void DnnSR::Load(const char* path)
{
    std::unique_ptr<ifstream> jsonfile(new ifstream(path));
    if(jsonfile->is_open()){
        *jsonfile >> config;
        logger.info("DnnSR::Load Success");
        logger.info("DnnSR::Load Config:"+config.dump());
    }else{
        std::cout<<"no config file "<<path<<std::endl;
    }
}

void DnnSR::Upsample(cv::Mat& src, cv::Mat& dst)
{
    logger.setOwnerFuncName(__func__);
    sr.upsample(src, dst);
}

