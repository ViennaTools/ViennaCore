#include <vcCudaBuffer.hpp>
#include <vcLaunchKernel.hpp>
#include <vcVectorType.hpp>

using namespace viennacore;

int main() {
  Logger::setLogLevel(LogLevel::DEBUG);

  DeviceContext context;
  context.create("../../../lib/ptx"); // relative to build directory
  const std::string moduleName = "test_kernel.ptx";
  context.addModule(moduleName);

	unsigned numResults = 10000;
	CudaBuffer resultBuffer;
	std::vector<Vec3Df> results(numResults, Vec3Df{0.0f, 0.0f, 0.0f});
	resultBuffer.allocUpload(results);

	auto add = Vec3Df{1.0f, 2.0f, -1.0f};
	CUdeviceptr d_data = resultBuffer.dPointer();
	void *kernel_args[] = {&add, &d_data, &numResults};

	LaunchKernel::launch(moduleName, "test_kernel", kernel_args, context);

	resultBuffer.download(results.data(), numResults);
	resultBuffer.free();
}
