#include <vcParameterFile.hpp>
#include <vcTestAsserts.hpp>

#include <sstream>

using namespace viennacore;

int main() {
  ParameterFile p;
  std::stringstream ss;
  ss << "par1=1.0\n";
  ss << "par2=2.0\n";
  ss << "par3=string_par\n";

  p.parseConfigStream(ss);

  auto par1 = p.get("par1");
  VC_TEST_ASSERT(par1 == 1.0);
  bool test_type = std::is_same_v<decltype(par1), double>;
  VC_TEST_ASSERT(test_type);

  auto par2 = p.get<float>("par2");
  VC_TEST_ASSERT(par2 == 2.0);
  test_type = std::is_same_v<decltype(par2), float>;
  VC_TEST_ASSERT(test_type);

  auto par3 = p.get<std::string>("par3");
  VC_TEST_ASSERT(par3 == "string_par");
}