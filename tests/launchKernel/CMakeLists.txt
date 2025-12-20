project(launchKernel LANGUAGES CXX)

if(VIENNACORE_USE_GPU)
  VIENNACORE_ADD_PTX(test_kernel "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu")

  add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp" ${generated_files})
  add_dependencies(${PROJECT_NAME} test_kernel)
  target_compile_definitions(${PROJECT_NAME} PRIVATE VIENNACORE_KERNELS_PATH_DEFINE=${VIENNACORE_PTX_DIR})

  target_link_libraries(${PROJECT_NAME} PRIVATE ViennaCore)

  add_dependencies(ViennaCore_Tests ${PROJECT_NAME})
  add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
endif()
