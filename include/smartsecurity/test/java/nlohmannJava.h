//
// Created by HP on 2025/9/26.
//

#ifndef SMARTSECURITYCORELIB_NLOHMANN_JAVA_H
#define SMARTSECURITYCORELIB_NLOHMANN_JAVA_H
#include "nlohmann/json.hpp"
#include <smartsecurity/data/output_json.h>
using nlohmann::json;

namespace java{

    void toJson(const char* str);
    void toJson(const char* str,data::json::OutputJson& outJson);
    void toJsonMessage(const char* str,data::json::OutputJson& outJson);


}

#endif //SMARTSECURITYCORELIB_NLOHMANN_JAVA_H
