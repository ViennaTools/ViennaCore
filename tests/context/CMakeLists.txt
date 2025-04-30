project(context LANGUAGES CXX)

if(VIENNACORE_USE_GPU)
  add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
  target_include_directories(${PROJECT_NAME} PRIVATE ${OptiX_INCLUDE} ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE ViennaCore ${VIENNACORE_GPU_LIBS})

  add_dependencies(ViennaCore_Tests ${PROJECT_NAME})
  add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
endif()
