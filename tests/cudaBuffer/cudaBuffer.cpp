#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>
#include <vcLogger.hpp>
#include <vcTestAsserts.hpp>

int main() {
  using namespace viennacore;
  Logger::setLogLevel(LogLevel::DEBUG);

  Context context;
  context.create(); // necessary to initialize CUDA

  CudaBuffer buffer;

  std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  buffer.allocUpload(data);

  std::vector<int> test(data.size());
  buffer.download(test.data(), data.size());

  for (int i = 0; i < data.size(); i++) {
    VC_TEST_ASSERT(test[i] == data[i]);
  }

  context.destroy();
}