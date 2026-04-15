cmake_minimum_required(VERSION 3.10)
project(SmartSecurityCoreLib)
set(CMAKE_CXX_STANDARD 20)
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


    include_directories(./include/ E:/CPPlib/SelfLib/onnxruntime-win-x64-gpu-1.20.1/include)
    link_directories(./lib/ E:/VScode/CUDA/lib/x64 E:/CPPlib/SelfLib/onnxruntime-win-x64-gpu-1.20.1/lib)


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

   src/cv/dnn/yolo/DnnYoloV5.cpp
   src/cv/dnn/yolo/DnnYoloBoot.cpp
   src/cv/dnn/ssd/DnnSSDcaffem.cpp
   src/cv/dnn/facenet/DnnFaceNet.cpp
   src/cv/dnn/facenet/DnnFaceNet512d.cpp
   src/cv/dnn/facenet/DnnArcFace.cpp
   src/cv/dnn/yunet/DnnYuNet.cpp
   src/cv/dnn/sr/DnnSR.cpp



   src/cv/main/edge/EdgeDetectionMain.cpp  #边缘检测主函数
   src/cv/main/dnn/DnnDetectionMain.cpp   #yolo检测主函数
   src/cv/main/dnn/DnnDetectionFaceMain.cpp

   src/cv/main/dnn/DnnYoloV5Main.cpp
   src/cv/main/dnn/sr/DnnSRMain.cpp
   src/cv/main/dnn/face/DnnSSDcaffemMain.cpp
   src/cv/main/dnn/face/DnnArcFaceMain.cpp
   src/cv/main/dnn/face/DnnYuNetMain.cpp
   src/cv/main/dnn/DnnYoloBootMian.cpp



   src/dlib/utlis/DlibUtils.cpp            #dlib工具类
   src/dlib/face/facialRecognition.cpp     #dlib人脸识别
   src/dlib/main/face/facialRecognitionMain.cpp
   src/cv/dnn/face/DnnFeatureFace.cpp
   src/cv/java/nlohmannJava.cpp
   src/cv/main/java/nlohmannJavaMain.cpp
   src/cv/main/data/DestroyMain.cpp

    )

    target_link_libraries(SmartSecurityCoreLibTest
            cudart
            cublas
            cusolver
            curand
            cudnn
            opencv_world4120
            dlib20.0.99_release_64bit_msvc1943
    )


    add_library(SSC-FaceLib SHARED
            src/cv/dnn/facenet/DnnFaceNet.cpp
            src/cv/dnn/facenet/DnnFaceNet512d.cpp

            src/cv/dnn/ssd/DnnSSDcaffem.cpp
            src/cv/main/dnn/face/DnnSSDcaffemMain.cpp

            src/cv/dnn/facenet/DnnArcFace.cpp
            src/cv/main/dnn/face/DnnArcFaceMain.cpp

            src/cv/dnn/yunet/DnnYuNet.cpp
            src/cv/main/dnn/face/DnnYuNetMain.cpp
    )
    add_library(SSC-YoloLib SHARED
            src/cv/dnn/yolo/DnnYoloV5.cpp
            src/cv/dnn/sr/DnnSR.cpp
            src/cv/main/dnn/DnnYoloV5Main.cpp
            src/cv/main/dnn/sr/DnnSRMain.cpp

    )

    target_link_libraries(SSC-FaceLib
            opencv_world4120
    )
     target_link_libraries(SSC-YoloLib
            opencv_world4120
    )



    add_executable(envT
            src/envT/envT.cpp
            src/cv/main/dnn/face/DnnArcFaceMain.cpp
            include/smartsecurity/cv/dnn/face/dnnYuNet.h
    )

    target_link_libraries(
            envT
            opencv_world4120
            onnxruntime
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
            src/cv/dnn/yolo/DnnYoloV5.cpp
            src/cv/dnn/yolo/DnnYoloBoot.cpp
            src/cv/dnn/facenet/DnnFaceNet.cpp
            src/cv/dnn/facenet/DnnFaceNet512d.cpp
            src/cv/dnn/facenet/DnnArcFace.cpp
            src/cv/dnn/ssd/DnnSSDcaffem.cpp
            src/cv/dnn/yunet/DnnYuNet.cpp
            src/cv/dnn/sr/DnnSR.cpp



            src/cv/main/edge/EdgeDetectionMain.cpp  #边缘检测主函数
            src/cv/main/dnn/DnnDetectionMain.cpp   #yolo检测主函数
            src/cv/main/dnn/DnnDetectionFaceMain.cpp

            src/cv/main/dnn/DnnYoloV5Main.cpp
            src/cv/main/dnn/sr/DnnSRMain.cpp
            src/cv/main/dnn/face/DnnSSDcaffemMain.cpp
            src/cv/main/dnn/face/DnnArcFaceMain.cpp
            src/cv/main/dnn/face/DnnYuNetMain.cpp


            src/cv/main/dnn/DnnYoloBootMian.cpp


#            src/dlib/utlis/DlibUtils.cpp            #dlib工具类
#            src/dlib/face/facialRecognition.cpp     #dlib人脸识别
#            src/dlib/main/face/facialRecognitionMain.cpp
            src/cv/dnn/face/DnnFeatureFace.cpp
            src/cv/main/data/DestroyMain.cpp

    )
    target_link_libraries(SmartSecurityCoreLibTest
            opencv_world
    )

endif()

