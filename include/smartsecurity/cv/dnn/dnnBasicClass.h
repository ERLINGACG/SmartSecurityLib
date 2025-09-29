//
// Created by HP on 2025/8/15.
//

#ifndef SMARTSECURITYCORELIB_DNN_BASIC_CLASS_H
#define SMARTSECURITYCORELIB_DNN_BASIC_CLASS_H
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <nlohmann/json.hpp>

#include "log/logger.h"
#include "log/dynamiclogger.h"
using std::string;
using nlohmann::json;
using std::ifstream;
using Ilogger::DynamicLogger;
namespace cv_dnn::dnnBasic{

    class DnnBasicClass {
        public:
           cv::dnn::Net net;
           json config;

           DynamicLogger logger;


           virtual   ~DnnBasicClass() =default;

           void Load(const char* path) {
               std::unique_ptr<ifstream> jsonfile(new ifstream(path));
               if(jsonfile->is_open()){
                     *jsonfile >> config;
               }else{
                     std::cout<<"no config file "<<path<<std::endl;
               }
           }

   };
}
#endif
