#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>
#include <vcLogger.hpp>
#include <vcTestAsserts.hpp>

int main() {
  using namespace viennacore;
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext();

  CudaBuffer buffer;

  std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  buffer.allocUpload(data);

  std::vector<int> test(data.size());
  buffer.download(test.data(), data.size());

  for (int i = 0; i < data.size(); i++) {
    VC_TEST_ASSERT(test[i] == data[i]);
  }

  double init = 42.0;
  CudaBuffer buffer2;
  buffer2.allocInit(10, init);
  std::vector<decltype(init)> test2(10);
  buffer2.download(test2.data(), 10);

  for (int i = 0; i < test2.size(); i++) {
    VC_TEST_ASSERT(test2[i] == 42.0);
  }

  buffer.free();
  buffer2.free();

  CudaBuffer buffer3 = buffer;
  VC_TEST_ASSERT(buffer3.dPointer() == buffer.dPointer());
  VC_TEST_ASSERT(buffer3.sizeInBytes == buffer.sizeInBytes);
  VC_TEST_ASSERT(buffer3.isRef);
}