project(sampling LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaCore)

add_dependencies(ViennaCore_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)

add_executable(samplingBenchmark "samplingBenchmark.cpp")
target_link_libraries(samplingBenchmark PRIVATE ViennaCore)