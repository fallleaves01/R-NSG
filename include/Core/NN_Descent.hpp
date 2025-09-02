#pragma once
#include <faiss/IndexNNDescent.h>
#include <faiss/MetricType.h>
#include <Graph/GraphIndex.hpp>
#include <PCH.hpp>
#include <Vector/VectorList.hpp>

namespace TDFANN {

namespace KNNG {

template <typename T>
Graph::GraphIndex<std::monostate> nn_descent(
    const Vector::VectorList<T>& dataset,
    size_t k) {
    Graph::GraphIndex<std::monostate> graph(dataset.size());
    std::vector<T> data(dataset.size() * dataset.dim());
    for (size_t i = 0; i < dataset.size(); i++) {
        std::ranges::copy(dataset[i], data.begin() + i * dataset.dim());
    }
    spdlog::info("start KNN train with size {}, dim {}", dataset.size(),
                 dataset.dim());
    faiss::IndexNNDescentFlat index(dataset.dim(), k);
    index.verbose = true;
    index.add(dataset.size(), data.data());
    spdlog::info("KNN final trained = {}, graph size = {}", index.is_trained,
                 index.nndescent.graph.size());
    for (size_t i = 0; i < dataset.size(); i++) {
        graph.add_neighbours(
            i, index.nndescent.final_graph | std::views::drop(i * k) |
                   std::views::take(k) | std::views::transform([&](size_t id) {
                       return Graph::GraphIndex<std::monostate>::Node{id, {}};
                   }));
    }
    return graph;
}

}  // namespace KNNG

}  // namespace TDFANN