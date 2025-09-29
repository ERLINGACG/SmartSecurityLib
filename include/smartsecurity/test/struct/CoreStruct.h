#ifndef CORESTRUCT_H
#define CORESTRUCT_H
#include <memory>
namespace smartsecurity {

namespace structs {
    struct CoreStruct{
            int width;
            int height;
            int size;
            std::unique_ptr<unsigned char[]> data;
            ~CoreStruct()=default;
        };
    }
}


#endif // CORESTRUCT_H