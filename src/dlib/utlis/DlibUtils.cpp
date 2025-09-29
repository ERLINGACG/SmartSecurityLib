//
// Created by HP on 2025/8/9.
//

#include "smartsecurity/test/dlib/utils/utils.h"

int dlib_utils::DlibUtils::clearOutputData(data::Output& outputData) {
    try{
        outputData.buffer = nullptr;
        outputData.size = 0;
    }catch(...){
        return -1;
    }
    return 0;
}

int dlib_utils::DlibUtils::deleteOutputData(data::Output* outputData) {
    try{
        if(outputData) {  // 添加空指针检查
            delete outputData;  // 使用delete而不是delete[]（除非明确是数组）
        }
    }catch(...){
        return -1;
    }
    return 0;
}
