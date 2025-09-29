//
// Created by HP on 2025/8/18.
//

#ifndef SMARTSECURITYCORELIB_J_VECTOR_H
#define SMARTSECURITYCORELIB_J_VECTOR_H

#include <memory>


namespace java{
    template<typename T>
    class JVector{
        private:
            int initialCapacity=10;
            int size=0;
            std::unique_ptr<T[]> data;
        public:
            explicit JVector(int size):size(size),data(std::make_unique<T[]>(size)){};
            ~JVector()=default;

    };
}
#endif //SMARTSECURITYCORELIB_J_VECTOR_H
