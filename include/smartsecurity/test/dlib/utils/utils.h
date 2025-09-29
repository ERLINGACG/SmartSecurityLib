//
// Created by HP on 2025/8/9.
//

#ifndef SMARTSECURITYCORELIB_UTILS_H
#define SMARTSECURITYCORELIB_UTILS_H
#include "smartsecurity/test/dlib/dlibexport.h"
namespace dlib_utils{
    class DlibUtils{
        public:
           static int clearOutputData(data::Output& outputData);
           static int deleteOutputData(data::Output* outputData);
    };
}


#endif //SMARTSECURITYCORELIB_UTILS_H
