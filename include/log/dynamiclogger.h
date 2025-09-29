//
// Created by HP on 2025/11/12.
//

#ifndef DYNAMIC_ICLOGGER_H
#define DYNAMIC_ICLOGGER_H
#include <chrono>
#include <format>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>
#include <mutex>

#include "asm/tidasm.h"
#ifdef _WIN32
     #define  I_GET_PID() iasm::GET_PID()
     #define  I_GET_TID() iasm::GET_TID()
#endif // _WIN32

using std::string;
using std::to_string;
using std::chrono::system_clock;
using std::put_time;
using std::mutex;
using std::lock_guard;
namespace Ilogger{

    namespace AnsiColor {
        const std::string RESET = "\033[0m";       // 重置颜色
        const std::string RED = "\033[31m";        // 红色
        const std::string GREEN = "\033[32m";      // 绿色
        const std::string YELLOW = "\033[33m";     // 黄色
        const std::string BLUE = "\033[34m";       // 蓝色
        const std::string MAGENTA = "\033[35m";    // 品红
        const std::string CYAN = "\033[36m";       // 青色
        const std::string WHITE = "\033[37m";      // 白色
        const auto BLUE_INFO = BLUE + " INFO " + RESET;



        const auto BLUE_TIME = BLUE + "[TIME]" + RESET;
        const auto BLUE_PID =  BLUE + "[PID:" + std::to_string(I_GET_PID()) + "]" + RESET;
    }

    class DynamicLogger{
        mutex Imtx=mutex();
        bool isLocked=true;
        struct Owner
        {
            string className;
            string funcName;
        } owner;
        static auto time_() {
            const auto now = system_clock::now();
            // 转换为time_t（秒级时间戳）
            auto t = system_clock::to_time_t(now);
            std::stringstream ss;
            tm tm_buf{};  // 定义 tm 结构体作为缓冲区（线程安全）
            tm* local_tm = &tm_buf;
            localtime_s(local_tm, &t);
            ss << put_time(local_tm, "[ %Y-%m-%d %H:%M:%S ]");
            return ss.str();
        }

    public:
        DynamicLogger()=default;
        ~DynamicLogger()=default;


        DynamicLogger& setOwnerName(const string& className){
            if(isLocked) {lock_guard lock(Imtx);}
            owner.className=className;
            return *this;
        }
        DynamicLogger& setOwnerFuncName(const string& funcName)
        {
            if(isLocked) {lock_guard lock(Imtx);}
            owner.funcName=funcName;
            return *this;
        }

        DynamicLogger& openLock(){
            isLocked=true;
            return *this;
        }

        DynamicLogger& closeLock(){
            isLocked=false;
            return *this;
        }

        DynamicLogger& info(const string& nsg){

            if(isLocked) {lock_guard lock(Imtx);}
            const string logMsg=time_()
                           +" INFO "
                           +"[PID:"+std::to_string(I_GET_PID())+"] "
                           +"[TID:"+std::to_string(I_GET_TID())+"]"
                           +"["+owner.className+"]"
                           +"["+owner.funcName+"]"
                           +nsg;
            // std::cout << logMsg << std::endl;
            printf("%s\n",logMsg.c_str());
            return *this;

        }

        DynamicLogger& error(const string& nsg){

            if(isLocked) {lock_guard lock(Imtx);}
            const string logMsg=AnsiColor::RED+time_()
                           +" ERROR "
                           +"[PID:"+std::to_string(I_GET_PID())+"] "
                           +"[TID:"+std::to_string(I_GET_TID())+"]"
                           +"["+owner.className+"]"
                           +"["+owner.funcName+"]"
                           +AnsiColor::RED+nsg+AnsiColor::RESET;
            printf("%s\n",logMsg.c_str());
            return *this;

        }
        DynamicLogger& warn(const string& nsg){

            if(isLocked) {lock_guard lock(Imtx);}
            const string logMsg=time_()
                           +" WARN "
                           +"[PID:"+std::to_string(I_GET_PID())+"] "
                           +"[TID:"+std::to_string(I_GET_TID())+"]"
                           +"["+owner.className+"]"
                           +"["+owner.funcName+"]"
                           +nsg;
            // std::cout << logMsg << std::endl;
            printf("%s\n",logMsg.c_str());
            return *this;

        }

        template <typename Func, typename... Args>
        DynamicLogger& time(Func&& func, Args&&... args) {
            if(isLocked) {lock_guard lock(Imtx);}

            auto invokeFunc=[&] {
                auto start=system_clock::now();
                try
                {
                      std::invoke(func, std::forward<Args>(args)...);
                      auto end=system_clock::now();
                      auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
                      return std::format("{}ms,success",duration.count());
                }catch (std::exception& e)
                {
                    error(e.what());
                    return std::format("{}",e.what());
                }
            };

            const string logMsg=AnsiColor::BLUE+time_()+
                           + " INFO "
                           +"[PID:"+std::to_string(I_GET_PID())+"] "
                           +"[TID:"+std::to_string(I_GET_TID())+"]"
                           +"["+owner.className+"]"
                           +"["+owner.funcName+"]"
                           +"[TIME]:" + ":" + invokeFunc() +AnsiColor::RESET
                           ;
            printf("%s\n",logMsg.c_str());
            return *this;
        }



    };
}
#endif //DYNAMIC_ICLOGGER_H
