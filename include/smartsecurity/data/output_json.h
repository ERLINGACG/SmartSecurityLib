//
// Created by HP on 2025/9/5.
//

#ifndef SMARTSECURITYCORELIB_OUTPUT_JSON_H
#define SMARTSECURITYCORELIB_OUTPUT_JSON_H
#include <memory>
namespace data::json{
    struct OutputJson{
        std::unique_ptr<char[]> json;
        int size;
    };
}
#endif //SMARTSECURITYCORELIB_OUTPUT_JSON_H
