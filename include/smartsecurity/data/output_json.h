//
// Created by HP on 2025/9/5.
//

#ifndef SMART_SECURITY_OUTPUT_JSON_H
#define SMART_SECURITY_OUTPUT_JSON_H
#include <memory>

#include "image_data.h"

namespace data::json{
    struct OutputJson{
        std::unique_ptr<char[]> json;
        int size;
    };

}
#endif
