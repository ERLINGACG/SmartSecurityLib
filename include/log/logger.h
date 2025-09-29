//
// Created by HP on 2025/11/11.
//

#ifndef LOGGER_H
#define LOGGER_H

#include <chrono>
#include <format>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>
#include <mutex>
// #ifdef _WIN32
//      #define  I_GET_PID() iasm::GET_PID()
//      #define  I_GET_TID() iasm::GET_TID()
// #endif // LOGGER_H_

#include "asm/tidasm.h"
using std::string;
using std::to_string;
using std::chrono::system_clock;
using std::put_time;
using std::mutex;
using std::lock_guard;
namespace Ilogger
{
    class Logger{
        // static mutex mtx;
        public:

             Logger()=default;
            ~Logger()=default;
        static string pid() { return std::to_string(iasm::GET_PID());};
        static string tid() { return std::to_string(iasm::GET_TID());};


       static string time() {
           const auto now = system_clock::now();
           // 转换为time_t（秒级时间戳）
           auto t = system_clock::to_time_t(now);
           std::stringstream ss;
           tm tm_buf{};  // 定义 tm 结构体作为缓冲区（线程安全）
           tm* local_tm = &tm_buf;
           localtime_s(local_tm, &t);
           ss << put_time(local_tm, "[ %Y-%m-%d %H:%M:%S ]");
           return ss.str();
        };
        // template <typename... Args>
        static void info(const string& className,const string& nsg){
            // lock_guard lock(mtx);
            std::cout << time()
                      << " INFO "
                      <<"[PID:"<<pid()<<"] "
                      <<"[TID:"<<tid()<<"]"
                      << "[" << className+"]" <<  nsg << std::endl;
        }


    };
   // mutex Logger::mtx=mutex();


}
#endif //LOGGER_H
