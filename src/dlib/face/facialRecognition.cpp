#include "smartsecurity/test/dlib/face/facialRecognition.h"


dlib_face::FacialRecognition::FacialRecognition(param::FaceNew* param){
    std::cout <<param->predictor_path << std::endl;
    std::cout <<param->recognition_Path << std::endl;
    try{
        deserialize(param->predictor_path) >> this->sp;      // 加载特征点模型
        deserialize(param->recognition_Path) >> this->net;  // 加载识别模型
        std::cout<<"加载成功"<<std::endl;
    }catch(std::exception& e){
        std::cerr << e.what() << std::endl;
    }
}
void dlib_face::FacialRecognition::getDetection(
    unsigned char *inputData_1, 
    int length,
    data::Output& outputData)
{
    cv::Mat img1 = cv::imdecode(cv::Mat(1, length, CV_8U, inputData_1), cv::IMREAD_COLOR);
    if(!img1.empty()){
        double scale = 1;
        cv::Mat resized;
        cv::resize(img1, resized, cv::Size(), scale, scale);
        std::cout<<"检测人脸"<<std::endl;
        frontal_face_detector detector = get_frontal_face_detector();

        dlib::matrix<dlib::rgb_pixel> face1;
        {
            dlib::cv_image<dlib::bgr_pixel> cimg(resized);
            auto faces = detector(cimg,0); //检测人脸区域
            if(faces.empty()){
                std::cout<<"未检测到人脸"<<std::endl;
                outputData.buffer = nullptr;
                outputData.size = 0;
                return ;
            }
                
            auto shape = sp(cimg, faces[0]); //提取人脸特征点
            dlib::extract_image_chip(
                    cimg,
                    dlib::get_face_chip_details(shape,150,0.25), face1
                    ); //提取人脸图像
        }
        std::cout<<"提取人脸图像成功"<<std::endl;
        try{
            // 在人脸提取前添加光照补偿
            dlib::matrix<dlib::rgb_pixel> normalized_face;
            dlib::equalize_histogram(face1, normalized_face);
            auto feature1 = net(normalized_face); //计算人脸特征值
            std::cout<<"计算人脸特征值成功"<<std::endl;
//            for(int i=0;i<128;i++){
//                std::cout<<feature1(i)<<std::endl;
//            }

            outputData.buffer = new unsigned char[128*sizeof(float)];  // 显式分配内存
            memcpy(outputData.buffer, &feature1(0), 128*sizeof(float));
            outputData.size = 128*sizeof(float); 

        }catch(std::exception& e){
            std::cout << e.what() << std::endl;
        }
       
    }
}
 dlib::matrix<float,0,1> dlib_face::FacialRecognition::byte_forMat(unsigned char* inputData_1, int length){
    if(length==128*sizeof(float)){
      auto* float_buffer = reinterpret_cast<float*>(inputData_1);
      dlib::matrix<float,0,1> reconstructed_feature = dlib::mat(float_buffer, 128, 1);

      if (reconstructed_feature.size() != 128) {
        std::cerr << "特征维度不匹配" << std::endl;
        return {};
      }
      return reconstructed_feature;
      
    }else{
        std::cerr << "输入数据长度不匹配" << std::endl;
    }
    return {};
}
double dlib_face::FacialRecognition::getDistance(
    unsigned char* inputData_1,
    int length_1,
    unsigned char* inputData_2,
    int length_2
){
    dlib::matrix<float,0,1> feature1 = byte_forMat(inputData_1, length_1);
    dlib::matrix<float,0,1> feature2 = byte_forMat(inputData_2, length_2);
    if(feature1.size() != 128 || feature2.size() != 128){
        std::cerr << "特征维度不匹配" << std::endl;
        return -1;
    }
    return dlib::length(feature1 - feature2);
}
