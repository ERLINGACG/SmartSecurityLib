#ifndef   __CV_EXPORT_H__
#define   __CV_EXPORT_H__
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/imgproc.hpp>
#include "smartsecurity/data/image_data.h"
#include "smartsecurity/data/output_json.h"
#include "smartsecurity/cv/edge/edgedetection.h"
#include "smartsecurity/cv/dnn/dnnDetection.h"
#include "smartsecurity/cv/dnn/dnnDetectionFace.h"
#include "smartsecurity/cv/utils/utils.h"
#include "smartsecurity/cv/param/dnnParam.h"
#include "smartsecurity/cv/dnn/dnnYoloV5.h"
#include "dnn/face/dnnSSDcaffem.h"
#include "dnn/face/dnnFaceNet.h"
#include "dnn/face/dnnFaceNet512d.h"
#include "dnn/face/dnnArcFace.h"
#include "dnn/face/dnnYuNet.h"
#include <vector>              // 新增：向量容器
#include <memory>              // 新增：智能指针
#include <chrono>              // 新增：时间测量
#include <iomanip>             // 新增：用于流控制符
#include <string>
#ifdef WIN32
    #define CORE_CV_API __declspec(dllexport)
#else
    #define CORE_CV_API  //  Unix/Linux 下为空
#endif

#define EXPORT_USE [[maybe_unused]]

#endif
