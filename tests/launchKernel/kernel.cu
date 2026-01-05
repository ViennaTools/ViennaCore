#include <cuda.h>
#include <cuda_runtime.h>

#include <vcVectorType.hpp>

extern "C" __global__ void test_kernel(viennacore::Vec3Di add,
                                       viennacore::Vec3Di *results,
                                       unsigned numResults) {
  using namespace viennacore;
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numResults; tidx += stride) {
    auto res = results[tidx] + add;
    results[tidx] = res;
  }
}
