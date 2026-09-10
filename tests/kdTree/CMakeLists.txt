project(kdTree LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaCore)

add_dependencies(ViennaCore_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)

add_executable(nfKDTree nfKDTree.cpp)
target_link_libraries(nfKDTree PRIVATE ViennaCore)
add_dependencies(ViennaCore_Tests nfKDTree)
add_test(NAME nfKDTree COMMAND $<TARGET_FILE:nfKDTree>)

add_executable(neighborSearch neighborSearch.cpp)
target_link_libraries(neighborSearch PRIVATE ViennaCore)
add_dependencies(ViennaCore_Tests neighborSearch)
add_test(NAME neighborSearch COMMAND $<TARGET_FILE:neighborSearch>)

add_executable(KDTreeBenchmark "KDTreeBenchmark.cpp")
target_link_libraries(KDTreeBenchmark PRIVATE ViennaCore)
