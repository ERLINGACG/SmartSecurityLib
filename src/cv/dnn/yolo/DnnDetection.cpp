#include "smartsecurity/cv/dnn/dnnDetection.h"
#include <iostream>
#include <fstream>
#include <map>
#include "opencv2/core/cuda.hpp"

dnnDetection::DnnDetectorYolo::DnnDetectorYolo(const char *jsonPath) {
    std::unique_ptr<ifstream> jsonfile(new ifstream(jsonPath));
    if(jsonfile->is_open()){
        nlohmann::json j;
        *jsonfile >> j;
         try{
             Load(j);
         }catch(const std::exception& e){
             std::cout<<"load config file failed"<<e.what()<<std::endl;
         }

    }else{
        std::cout<<"no config file "<<jsonPath<<std::endl;
    }
}

void dnnDetection::DnnDetectorYolo::Load(nlohmann::json &j) {

    for(const auto& e : j["classNames"]){
        this->classNames.emplace_back(e);
    }
    this->net= readNetFromONNX(string(j["modelPath"]));
    if(this->net.empty()){
        std::cout<<"load model failed"<<std::endl;
    }
    this->net.setPreferableBackend(DNN_BACKEND_CUDA);
    this->net.setPreferableTarget(DNN_TARGET_CUDA);
    this->confThreshold=j["confThreshold"];
    this->nmsThreshold=j["nmsThreshold"];

    cout<<"classNames size:"<<this->classNames.size()<<endl;
    cout<<"modelPath:"<<j["modelPath"]<<endl;
    cout<<"confThreshold:"<<this->confThreshold<<endl;
    cout<<"nmsThreshold:"<<this->nmsThreshold<<endl;
}
dnnDetection::DnnDetectorYolo::DnnDetectorYolo(
    const char* Yolo_path,bool isCUDA,double confThreshold,double nmsThreshold
)
{
    try{
        this->net = readNetFromONNX(Yolo_path);

      if(isCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            this->net.setPreferableBackend(DNN_BACKEND_CUDA);
            this->net.setPreferableTarget(DNN_TARGET_CUDA);
            this->confThreshold=confThreshold;
            this->nmsThreshold=nmsThreshold;
            std::cout << "[CUDA] 设备初始化完成（" << cv::cuda::getDevice() << "）" << std::endl;
            std::cout<<"参数预设："<<std::endl;
            std::cout<<"置信度阈值："<<this->confThreshold<<std::endl;
            std::cout<<"非极大值抑制阈值："<<this->nmsThreshold<<std::endl;

        } else {
            this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
            this->net.setPreferableTarget(DNN_TARGET_CPU);
            this->confThreshold=confThreshold;
            this->nmsThreshold=nmsThreshold;
            std::cout << "[CPU] 初始化完成" << std::endl;
        }
        std::unique_ptr<ifstream> jsonfile(new ifstream("lib/x64/debug/config/classNames.json"));
        if(jsonfile->is_open()){
            nlohmann::json j;
            *jsonfile >> j;
            for(const auto& e : j){
                this->classNames.emplace_back(e);
            }
            for(const auto& e : this->classNames){
                std::cout<<e<<std::endl;
            }
        }else{
            std::cout<<"classNames.json file is not open"<<std::endl;
        }

        
    }catch(...){
        std::cout<<"加载模型失败"<<std::endl;
    }
    
}
void dnnDetection::DnnDetectorYolo::inputImage(unsigned char *inputData, int size, Mat &orgImage) {
    if(inputData==nullptr || size<=0){
        std::cout<<"inputImage failed,inputData is null or size is 0"<<std::endl;
        return;
    }
    std::vector<unsigned char> buf(inputData, inputData + size); //构造临时缓冲区
    orgImage = cv::imdecode(buf, cv::IMREAD_COLOR);  // 修正参数为正确解码标志
}
void dnnDetection::DnnDetectorYolo::outputImage(cv::Mat& inputImg,
                                                std::vector<int>& class_ids,//分类类别索引
                                                std::vector<float>& confidences,//置信度
                                                std::vector<cv::Rect>& boxes,
                                                std::vector<int>& nms_result,
                                                data::ImageData& OutputData,
                                                data::json::OutputJson& json
                                                )
{
    auto dataMap = std::map<std::string,int>();
    for(int index : nms_result){
        cv::Rect box = boxes[index];
        if(this->classNames[class_ids[index]]!="dining table"){

            cv::rectangle(inputImg, box, cv::Scalar(0, 255, 0), 2);
            cv::putText(inputImg, this->classNames[class_ids[index]] + ": " + to_string(confidences[index]), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            std::cout<<this->classNames[class_ids[index]]<<":"<<confidences[index]<<std::endl;
            // 直接增加计数，如果键不存在会自动创建并初始化为0
            dataMap[this->classNames[class_ids[index]]]++;
        }
    }
    std::string  dataJson;
    // 打印每个类别的检测数量
    if(!dataMap.empty()){

        for (const auto& pair : dataMap) {
            std::cout << pair.first << ": " << pair.second << std::endl;
            dataJson += "\"" + pair.first + "\": " + std::to_string(pair.second) + ",";
        }
        dataJson.pop_back(); // 移除最后一个逗号
        dataJson = "{" + dataJson + "}";
        std::cout<<dataJson<<std::endl;

        auto buffer = std::make_unique<char[]>(dataJson.size() + 1);
        std::copy(dataJson.begin(), dataJson.end(), buffer.get());
        buffer[dataJson.size()] = '\0';
        json.json = std::move(buffer);
        json.size = static_cast<int>(dataJson.size());
    }else{
        std::cout<<"没有检测到目标"<<std::endl;
        dataJson = "{}";
    }
    std::vector<int> params_ {cv::IMWRITE_WEBP_QUALITY, 95};
    std::vector<unsigned char> encoded;
    cv::imencode(".jpeg", inputImg, encoded, params_);
    std::cout<<"编码完成"<<std::endl;
    OutputData.size =  static_cast<int>(encoded.size());
    OutputData.width = inputImg.cols;
    OutputData.height = inputImg.rows;
    OutputData.channels = 3;
    OutputData.data = std::make_unique<unsigned char[]>(encoded.size());
    memcpy(OutputData.data.get(), encoded.data(), encoded.size());

}
void dnnDetection::DnnDetectorYolo::SetBlob(cv::Mat& blob, cv::Mat& inputImg) {
    blob=blobFromImage(
            inputImg, 1/255.0, Size(640,640),
            Scalar(0, 0, 0), true, false);
    this->net.setInput(blob); // 设置输入

}
void dnnDetection::DnnDetectorYolo::Forward(std::vector<Mat>& output_mat) {
    vector<string> outputs_name = this->net.getUnconnectedOutLayersNames(); // 获取输出层名称
    this->net.forward(output_mat, outputs_name); // 前向传播
    std::cout<<"前向传播完成"<<std::endl;
    if(output_mat.empty() || !output_mat[0].data){
        std::cerr << "模型无有效输出" << std::endl;
        return;
    }
    std::cout << "输出矩阵维度: " << output_mat[0].size << std::endl; // 打印维度信息


}

void dnnDetection::DnnDetectorYolo::ProcessResults(
        cv::Mat& inputImg,
        std::vector<Mat>& output_mat,
        std::vector<int>& class_ids,//分类类别索引
        std::vector<float>& confidences,//置信度
        std::vector<cv::Rect>& boxes,//边框坐标信息
        std::vector<int>& nms_result
        )
{
    auto* data = (float*)output_mat[0].data;
    float x_factor = static_cast<float>(inputImg.cols) / 640.0f;
    float y_factor = static_cast<float>(inputImg.rows) / 640.0f;

    int rows = 25200;
    for (int i = 0; i < rows; i++) {
        float confidence = data[4]; //置信度
//        std::cout << "检测框" << i << " 置信度: " << confidence << std::endl;  // 新增
        if (confidence > this->confThreshold) { // 过滤置信度低的目标
            float* classes_scores = data + 5;    //分类分数
            cv::Mat scores(1, static_cast<int>(this->classNames.size()), CV_32FC1, classes_scores); //转换为Mat
            cv::Point class_id;     //类别索引
            double max_class_score; //最大分数
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);   //获取最大分数的类别索引
            if (max_class_score > 0.25) { // 过滤分数低的目标
                confidences.push_back(confidence);//置信度
                class_ids.push_back(class_id.x);  //类别索引
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor); //左上角x坐标
                int top = int((y - 0.5 * h) * y_factor); //左上角y坐标
                int width = int(w * x_factor);           //宽度
                int height = int(h * y_factor);          //高度
                boxes.emplace_back(left, top, width, height);
            }
        }
        data += 85; //跳过85个元素，到下一个检测框
    }
    std::cout<<"检测框数量: "<<boxes.size()<<std::endl;
    cv::dnn::NMSBoxes(
            boxes, confidences,
            static_cast<float>(this->confThreshold),
            static_cast<float>(this->nmsThreshold), nms_result
    ); //非极大值抑制



}
void dnnDetection::DnnDetectorYolo::DetectImage_3(
        unsigned char *inputData,
        int size, data::ImageData &OutputData,
        data::json::OutputJson &json
) {
    std::lock_guard<std::mutex> lock(m_mutex); //确保线程安全
    //预处理
    cv::Mat orgImage;
    cv::Mat blob;
    std::vector<Mat> output_mat;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> nms_result;

    auto t1 = std::chrono::system_clock::now();
    inputImage(inputData,size,orgImage);
    auto t2 = std::chrono::system_clock::now();
    std::cout<<"输入编码完成"<<endl;

    auto t3 = std::chrono::system_clock::now();
    cv_utils::ImageUtils::resizeWithPadding_mat(orgImage,orgImage,640,640); //原地处理
    auto t4 = std::chrono::system_clock::now();
    std::cout<<"预处理完成"<<endl;
    auto t5 = std::chrono::system_clock::now();
    SetBlob(blob,orgImage);
    auto t6 = std::chrono::system_clock::now();
    std::cout<<"blob设置完成"<<endl;
    Forward(output_mat);
    auto t7 = std::chrono::system_clock::now();
    std::cout<<"前向传播完成"<<std::endl;
    ProcessResults(orgImage,output_mat,class_ids,confidences,boxes,nms_result);
    auto t8 = std::chrono::system_clock::now();
    std::cout<<"推理完成"<<std::endl;
    outputImage(orgImage,
                class_ids,//分类类别索引
                confidences,//置信度
                boxes,    //边框坐标信息
                nms_result,    //非极大值抑制结果
                OutputData,
                json);
    std::cout<<"输出编码完成"<<endl;

    auto fmt_time = [](auto time) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
        return std::to_string(us/1000.0) + "ms";
    };
    std::cout<<"输入编码耗时: "<<fmt_time(t2-t1)
             <<" | 预处理耗时: "<<fmt_time(t4-t3)
             <<" | blob设置耗时: "<<fmt_time(t6-t5)<<" | 前向传播耗时: "<<fmt_time(t7-t6)
             <<" | 推理耗时: "<<fmt_time(t8-t7)
             <<" | 总耗时: "<<fmt_time(t8-t1)<<std::endl;
}

void dnnDetection::DnnDetectorYolo::DetectImage(
        unsigned char* inputData, int size,
        data::ImageData& OutputData,
        data::json::OutputJson& json
){
    std::lock_guard<std::mutex> lock(m_mutex); // 加锁
    std::cout<<"开始检测"<<std::endl;
    auto start_total = std::chrono::system_clock::now(); // 总开始时间
    auto t_decode_start = std::chrono::system_clock::now();
    if(inputData!=nullptr){

       std::vector<unsigned char> buf(inputData, inputData + size); //构造临时缓冲区
       cv::Mat inputImg = cv::imdecode(buf, cv::IMREAD_COLOR);  // 修正参数为正确解码标志
       auto t_decode_end = std::chrono::system_clock::now();
       std::cout<<"色彩转换完成"<<std::endl;
       auto t1=std::chrono::system_clock::now(); 
       
       auto t_resize_start = std::chrono::system_clock::now();
       cv_utils::ImageUtils::resizeWithPadding_mat(inputImg,inputImg,640,640);
       auto t_resize_end = std::chrono::system_clock::now();
       std::cout<<"预处理完成"<<std::endl;

       auto t_blob_start = std::chrono::system_clock::now();
       Mat blob=blobFromImage(inputImg, 1/255.0, Size(640,640), Scalar(0, 0, 0), true, false); 
       this->net.setInput(blob); // 设置输入
       auto t_blob_end = std::chrono::system_clock::now();

       vector<string> outputs_name = this->net.getUnconnectedOutLayersNames(); // 获取输出层名称
       auto t_forward_start = std::chrono::system_clock::now();
       vector<Mat> output_mat;
       this->net.forward(output_mat, outputs_name); // 前向传播
       auto t_forward_end = std::chrono::system_clock::now();
       std::cout<<"前向传播完成"<<std::endl;
        if(output_mat.empty() || !output_mat[0].data){
            std::cerr << "模型无有效输出" << std::endl;
            return;
        }
        std::cout << "输出矩阵维度: " << output_mat[0].size << std::endl; // 打印维度信息

       auto t_postprocess_start = std::chrono::system_clock::now();
       auto* data = (float*)output_mat[0].data;
       float x_factor = inputImg.cols / 640.0f;
       float y_factor = inputImg.rows / 640.0f;

       //yolov5s模型的输出大小为[1,25200.85]
       //yolov5-A模型的输出大小为[1,25200.11]
        const int dimensions = 11;
        const int rows = 25200;
            
        vector<int> class_ids;//分类类别索引
        vector<float> confidences;//置信度
        vector<cv::Rect> boxes;//边框坐标信息
        std::cout<<"当前任务：检测"<<std::endl;
        std::cout<<"输出层名称："<<outputs_name[0]<<std::endl;

        for (int i = 0; i < rows; i++) {
            float confidence = data[4]; //置信度
            if (confidence > this->confThreshold) { // 过滤置信度低的目标
                  float* classes_scores = data + 5;    //分类分数
                  cv::Mat scores(1, static_cast<int>(this->classNames.size()), CV_32FC1, classes_scores); //转换为Mat
                  cv::Point class_id;     //类别索引  
                  double max_class_score; //最大分数  
                  cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);   //获取最大分数的类别索引
                  if (max_class_score > 0.25) { // 过滤分数低的目标
                     confidences.push_back(confidence);//置信度
				     class_ids.push_back(class_id.x);  //类别索引
                     float x = data[0];
                     float y = data[1];
                     float w = data[2];
                     float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor); //左上角x坐标
				    int top = int((y - 0.5 * h) * y_factor); //左上角y坐标
				    int width = int(w * x_factor);           //宽度
				    int height = int(h * y_factor);          //高度
                    boxes.push_back(cv::Rect(left, top, width, height));
                  }
            }
            data += 85; //跳过85个元素，到下一个检测框
        }
        auto t_postprocess_end = std::chrono::system_clock::now();

        std::vector<int> nms_result; //nms结果
//        std::cout<<this->confThreshold<<std::endl;
//        std::cout<<this->nmsThreshold<<std::endl;

	    cv::dnn::NMSBoxes(
            boxes, confidences,
            static_cast<float>(this->confThreshold),
            static_cast<float>(this->nmsThreshold), nms_result
        ); //非极大值抑制
        
        auto t_draw_start = std::chrono::system_clock::now();
        auto dataMap = std::map<std::string,int>();
        for(int i = 0; i < nms_result.size(); i++){
            int index = nms_result[i];
            cv::Rect box = boxes[index];
            if(this->classNames[class_ids[index]]!="dining table"){
                    
                    cv::rectangle(inputImg, box, cv::Scalar(0, 255, 0), 2);
                    cv::putText(inputImg, this->classNames[class_ids[index]] + ": " + to_string(confidences[index]), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                    std::cout<<this->classNames[class_ids[index]]<<":"<<confidences[index]<<std::endl;
                // 直接增加计数，如果键不存在会自动创建并初始化为0
                dataMap[this->classNames[class_ids[index]]]++;
//                confidencesMap[this->classNames[class_ids[index]]].push_back(confidences[index]);
            }
        }
        std::string  dataJson;
        // 打印每个类别的检测数量
        if(!dataMap.empty()){

            for (const auto& pair : dataMap) {
                std::cout << pair.first << ": " << pair.second << std::endl;
                dataJson += "\"" + pair.first + "\": " + std::to_string(pair.second) + ",";
            }
            dataJson.erase(dataJson.size() - 1);
            dataJson = "{" + dataJson + "}";
            std::cout<<dataJson<<std::endl;

            auto buffer = std::make_unique<char[]>(dataJson.size() + 1);
            std::copy(dataJson.begin(), dataJson.end(), buffer.get());
            buffer[dataJson.size()] = '\0';
            json.json = std::move(buffer);
            json.size = static_cast<int>(dataJson.size());
        }else{
            dataJson = "{}";
        }




        auto t_draw_end = std::chrono::system_clock::now();
        std::cout<<"绘制结果完成"<<std::endl;

        auto t_encode_start = std::chrono::system_clock::now();
        std::vector<int> params_ {cv::IMWRITE_WEBP_QUALITY, 95};
        std::vector<unsigned char> encoded;
        cv::imencode(".jpeg", inputImg, encoded, params_);
        
        std::cout<<"编码完成"<<std::endl;
        OutputData.size =  static_cast<int>(encoded.size());
        OutputData.width = inputImg.cols;
        OutputData.height = inputImg.rows;
        OutputData.channels = 3;
        OutputData.data = std::make_unique<unsigned char[]>(encoded.size()); 
        memcpy(OutputData.data.get(), encoded.data(), encoded.size());
        auto t_encode_end = std::chrono::system_clock::now();
        std::cout<<"检测完成"<<std::endl;
         // 添加性能日志输出（放在函数末尾）
       auto fmt_time = [](auto time) {
           auto us = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
           return std::to_string(us/1000.0) + "ms";
       };

       std::cout << std::fixed << std::setprecision(2);
       std::cout << "[YOLO Perf] 解码:" << fmt_time(t_decode_end - t_decode_start)
                 << " | 缩放:" << fmt_time(t_resize_end - t_resize_start)
                 << " | Blob生成:" << fmt_time(t_blob_end - t_blob_start)
                 << " | 推理:" << fmt_time(t_forward_end - t_forward_start)
                 << " | 后处理:" << fmt_time(t_postprocess_end - t_postprocess_start)
                 << " | 绘制:" << fmt_time(t_draw_end - t_draw_start)
                 << " | 编码:" << fmt_time(t_encode_end - t_encode_start)
                 << " | 总耗时:" << fmt_time(std::chrono::system_clock::now() - start_total)
                 << std::endl;

    }else{
        std::cout<<"输入数据为空"<<std::endl;
        return;
    }
}

