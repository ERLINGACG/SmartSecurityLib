#ifndef _IMAGE_DATA_H_
#define _IMAGE_DATA_H_
#include <memory>
namespace data{
    struct ImageData{
        int width;
        int height;
        int channels;
        int size;
        std::unique_ptr<unsigned char[]> data;
       
        ~ImageData()=default;
    };

    struct FaceFeatureByte{
        int dataSize;
        int faceNum;
        std::unique_ptr<unsigned char[]> data;
    };

};

#endif