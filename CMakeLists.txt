cmake_minimum_required(VERSION 3.10)
project(SmartSecurityCoreLib)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (WIN32)
if(MSVC)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "86")  # 对应3050TI的Ampere架构
    add_definitions(-DUNICODE -D_UNICODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast /openmp")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /utf-8")
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()


include_directories(./include/)
link_directories(./lib/ E:/VScode/CUDA/lib/x64)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/debug)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release)
endif()

add_library(SmartSecurityCoreLibTest SHARED
   src/cv/edge/EdgeDetection.cpp  #边缘检测
   src/cv/dnn/yolo/DnnDetection.cpp   #yolo检测
   src/cv/utils/Utils.cpp         #工具类
   src/cv/dnn/face/DnnDetectionFace.cpp

   src/cv/main/edge/EdgeDetectionMain.cpp  #边缘检测主函数
   src/cv/main/dnn/DnnDetectionMain.cpp   #yolo检测主函数
   src/cv/main/dnn/DnnDetectionFaceMain.cpp


   src/dlib/utlis/DlibUtils.cpp            #dlib工具类
   src/dlib/face/facialRecognition.cpp     #dlib人脸识别
   src/dlib/main/face/facialRecognitionMain.cpp
   src/cv/dnn/face/DnnFeatureFace.cpp
   src/cv/java/nlohmannJava.cpp
   src/cv/main/java/nlohmannJavaMain.cpp


)
#add_executable(main
#        src/cv/test/test.cpp
#)



target_link_libraries(SmartSecurityCoreLibTest
        cudart
        cublas
        cusolver
        curand
        cudnn
        opencv_world4120
        dlib20.0.99_release_64bit_msvc1943
)

target_link_options(SmartSecurityCoreLibTest PRIVATE /LTCG)
target_compile_definitions(SmartSecurityCoreLibTest PRIVATE "SMARTSECURITY_CORE_EXPORTS")
endif ()

if(UNIX)
    include_directories(./include/)
    link_directories(./lib/)

    add_library(SmartSecurityCoreLibTest SHARED
            src/cv/edge/EdgeDetection.cpp  #边缘检测
            src/cv/dnn/yolo/DnnDetection.cpp   #yolo检测
            src/cv/utils/Utils.cpp         #工具类
            src/cv/dnn/face/DnnDetectionFace.cpp

            src/cv/main/edge/EdgeDetectionMain.cpp  #边缘检测主函数
            src/cv/main/dnn/DnnDetectionMain.cpp   #yolo检测主函数
            src/cv/main/dnn/DnnDetectionFaceMain.cpp


#            src/dlib/utlis/DlibUtils.cpp            #dlib工具类
#            src/dlib/face/facialRecognition.cpp     #dlib人脸识别
#            src/dlib/main/face/facialRecognitionMain.cpp
            src/cv/dnn/face/DnnFeatureFace.cpp
    )
    target_link_libraries(SmartSecurityCoreLibTest
            opencv_world
    )

endif()

