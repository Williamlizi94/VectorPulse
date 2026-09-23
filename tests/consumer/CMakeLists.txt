cmake_minimum_required(VERSION 3.20)
project(VectorPulseConsumer LANGUAGES CXX)

find_package(VectorPulse REQUIRED)
add_executable(vectorpulse_consumer main.cpp)
target_link_libraries(vectorpulse_consumer PRIVATE VectorPulse::vectorpulse)

# C++20 and include paths must come from the installed imported target.
enable_testing()
add_test(NAME consumer_search COMMAND vectorpulse_consumer)
