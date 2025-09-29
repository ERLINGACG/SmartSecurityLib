#ifndef _DLLEXPORT_H_
#define _DLLEXPORT_H_
#define DLIB_JPEG_SUPPORT
#define DLIB_PNG_SUPPORT
//#define DLIB_DNN_USE_CUDA
#define DLIB_USE_CUDA
#define DLIB_DNNC_USE_CUDA
#include "dlib/dnn.h"
#include "dlib/gui_widgets.h"
#include "dlib/clustering.h"
#include "dlib/string.h"
#include "dlib/image_io.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/opencv.h"
#include "dlib/image_processing/render_face_detections.h"

#include "smartsecurity/test/dlib/temp/cnn_temp.h"
#include "smartsecurity/test/dlib/param/faceNew.h"
#include "smartsecurity/test/dlib/face/facialRecognition.h"
#include "smartsecurity/test/dlib/data/output.h"
#include "smartsecurity/test/dlib/utils/utils.h"

#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include "opencv2/imgcodecs.hpp"  // 包含imdecode函数声明
#include "opencv2/opencv.hpp"     // 或直接包含核心头文件
#define CORE_DLIB_API __declspec(dllexport)

#endif // _DLLEXPORT_H_