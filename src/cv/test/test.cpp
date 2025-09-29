#include <iostream>
#include <map>
#include "nlohmann/json.hpp"

int main() {
    std::string s = R"([{"1758790014974":[{"laptop": 1,"person": 1}],"1758790015043":[{"person": 1}]}])";
    nlohmann::json j = nlohmann::json::parse(s);
    int index = 0;
    std::string  key,value;
    std::map<std::string,int> map;
    for(const auto& sz: j.items().begin().value()){
        for (auto& i: sz.items()) {
            for (auto& it: i.value().items()) {
                std::cout << it.key() << " " << it.value() << std::endl;
                map[it.key()] += it.value().get<int>();
            }
//
        }
    }
    for (auto& it: map) {
        std::cout << it.first << " " << it.second << std::endl;
    }
    std::cout << key << " " << value << std::endl;
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
