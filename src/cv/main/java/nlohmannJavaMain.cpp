//
// Created by HP on 2025/9/26.
//
#include <iostream>
#include "smartsecurity/test/java/nlohmannJava.h"
#include "smartsecurity/cv/cvexport.h"


extern "C" {
    EXPORT_USE CORE_CV_API void ToJson(const char *str) {
        java::toJson(str);
    }
    EXPORT_USE CORE_CV_API void ToJsonOutput(const char *str,data::json::OutputJson &outJson) {
        try{
            java::toJson(str, outJson);
        }catch (const std::exception& e){
            std::cerr<<"ToJsonOutput error: "<<e.what()<<std::endl;
        }
    }
    EXPORT_USE CORE_CV_API void ToJsonTest(const char *str,data::json::OutputJson &outJson) {
        try{
            java::toJsonMessage(str, outJson);
        }catch (const std::exception& e){
            std::cerr<<"ToJsonMessage error: "<<e.what()<<std::endl;
        }
    }

}