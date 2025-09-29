//
// Created by HP on 2025/9/26.
//
#include "smartsecurity/test/java/nlohmannJava.h"
#include <iostream>

void java::toJson(const char* str){
    nlohmann::json j = nlohmann::json::parse(str);
    int index = 0;
    std::string  key,value;
    std::map<std::string,int> map;
    for (const auto& i1: j.items()) {
        for (const auto& i2: i1.value().items()) {
            for (const auto& i3: i2.value().items()) {
                for (const auto& i4: i3.value().items()) {
                    map[i4.key()] += i4.value().get<int>();
                }
            }
        }
    }

    for (auto& it: map) {
        std::cout << it.first << " " << it.second << std::endl;
    }
}
void java::toJson(const char *str, data::json::OutputJson &outJson) {

    nlohmann::json j = nlohmann::json::parse(str);
    if (j.empty()){
        std::string  default_="{}";
        outJson.size = (int)default_.size();
        outJson.json = std::make_unique<char[]>(outJson.size);
        std::strcpy(outJson.json.get(), default_.c_str());
    }else{
        std::string  key,value;
        std::unordered_map<std::string,int> map;
        for (const auto& i1: j.items()) {
            for (const auto& i2: i1.value().items()) {
                for (const auto& i3: i2.value().items()) {
                    for (const auto& i4: i3.value().items()) {
                        map[i4.key()] += i4.value().get<int>();
                    }
                }
            }
        }
        nlohmann::json j2;
        for (auto& it: map) {
            j2[it.first] = it.second;
        }
        std::string jsonStr = j2.dump(4);
        outJson.size = (int)jsonStr.size();
        outJson.json = std::make_unique<char[]>(outJson.size);
        std::strcpy(outJson.json.get(), jsonStr.c_str());
    }
}
void java::toJsonMessage(const char *str, data::json::OutputJson &outJson) {
    long long nextTime=0;
    long long diffTime=0;
    std::map<std::string,int> classMap;
    nlohmann::json j = nlohmann::json::parse(str);
    for (const auto& [key,value]: j.items()){
      for (const auto& [key2,value2]: value.items()){

              long long time = std::stoll(key2);  //计算时间差
              diffTime += time -nextTime;
              nextTime=time;
//              if (diffTime >=1000*120) {
                  for (const auto& item: value2.items()){
                      for (const auto& [key3,value3]: item.value().items()){

                          if(classMap[key3] == 0){
                              classMap[key3] = 1;
                          }
                          if(classMap[key3] <= value3.get<int>()){
                              classMap[key3] = value3.get<int>();
                          }
                          if(diffTime >=1000*120){
                              classMap[key3] += value3.get<int>();
                              diffTime=0;
                          }

                      }
                      std::cout << item.key() << " " << item.value() << std::endl;
                  }

//               }
              std::cout << key2 << " " << value2 << " " << diffTime << std::endl;
      }
    }
    for (auto& it: classMap) {
        std::cout << it.first << " " << it.second << std::endl;
    }
}
