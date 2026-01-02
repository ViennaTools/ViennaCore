#include <vcCudaBuffer.hpp>
#include <vcLaunchKernel.hpp>
#include <vcVectorType.hpp>

using namespace viennacore;

int main() {
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext();
  const std::string moduleName = "test_kernel.ptx";
  context->addModule(moduleName);

  unsigned numResults = 10000;
  CudaBuffer resultBuffer;
  std::vector<Vec3Di> results(numResults, Vec3Di{0, 0, 0});
  resultBuffer.allocUpload(results);

  auto add = Vec3Di{1, 2, -1};
  CUdeviceptr d_data = resultBuffer.dPointer();
  void *kernel_args[] = {&add, &d_data, &numResults};

  LaunchKernel::launch(moduleName, "test_kernel", kernel_args, *context);

  resultBuffer.download(results.data(), numResults);
  resultBuffer.free();

  // Verify results
  bool allCorrect = true;
  for (unsigned i = 0; i < numResults; ++i) {
    if (results[i] != add) {
      allCorrect = false;
      std::cout << "Error at index " << i << ": got (" << results[i] << ")\n";
      break;
    }
  }

  if (allCorrect) {
    std::cout << "All results are correct.\n";
    return 0;
  } else {
    std::cout << "There were errors in the results.\n";
    return 1;
  }
}
