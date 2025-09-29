#ifndef SMARTSECURITY_DLIB_DATA_OUTPUT_H_
#define SMARTSECURITY_DLIB_DATA_OUTPUT_H_
#include "smartsecurity/test/dlib/dlibexport.h"

    namespace data{
        struct Output {
            unsigned char* buffer;  // 原始字节指针
            int size;            // 数据长度
        };
    } 


#endif
