#include "vectorpulse/vector_index.h"

#include <iostream>
#include <vector>

int main() {
    vectorpulse::VectorIndex index{3};
    index.add("east", {1.0F, 0.0F, 0.0F});
    index.add("diagonal", {1.0F, 1.0F, 0.0F});
    index.add("north", {0.0F, 1.0F, 0.0F});

    const std::vector<float> query{1.0F, 0.0F, 0.0F};
    std::cout << index.size() << " vectors, dimension " << index.dimension() << '\n';
    for (const auto& result : index.search(query, 2)) {
        std::cout << result.id << ": " << result.score << '\n';
    }
}
