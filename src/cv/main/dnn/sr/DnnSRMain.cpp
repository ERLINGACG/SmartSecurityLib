//
// Created by HP on 2025/11/24.
//
#include "smartsecurity/cv/cvexport.h"
using cv_sr::DnnSR;

extern  "C"{
    EXPORT_USE  CORE_CV_API DnnSR* DnnSRCreate(const char* path){
        return new DnnSR(path);
    }
}
