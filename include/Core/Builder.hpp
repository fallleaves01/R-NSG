#pragma once
#include <PCH.hpp>
#include <Vector/VectorList.hpp>

namespace TDFANN {

template <typename T>
class Builder {
   public:
    Builder();

   private:
    Vector::VectorList<T> vector_list;  // 向量列表
};

}  // namespace TDFANN