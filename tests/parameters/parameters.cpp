#include <vcTestAsserts.hpp>
#include <vcUtil.hpp>

#include <sstream>

using namespace viennacore;

int main() {
  util::Parameters p;
  std::stringstream ss;

  ss << "par1=1.0\n";
  ss << "par2= 2.0\n";
  ss << "par3 =string_par\n";
  ss << "par4 = 4\n";
  ss << "integers = 1, -2, +3 # a comment, with a comma\n";
  ss << "doubles = 1.5,2.5,-3e-2\n";
  ss << "strings = first, second_value,\tthird.value\n";
  ss << "bools = true,\tfalse, true\n";
  ss << "invalid = 1, nope, 3\n";
  ss << "empty_item = 1, , 3\n";
  ss << "trailing_comma = 1, 2,\n";

  p.readConfigStream(ss);

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

  auto par4 = p.get<int>("par4");
  VC_TEST_ASSERT(par4 == 4);
  test_type = std::is_same_v<decltype(par4), int>;
  VC_TEST_ASSERT(test_type);

  const auto integers = p.get<std::vector<int>>("integers");
  VC_TEST_ASSERT((integers == std::vector<int>{1, -2, 3}));

  const auto doubles = p.get<std::vector<double>>("doubles");
  VC_TEST_ASSERT((doubles == std::vector<double>{1.5, 2.5, -0.03}));

  const auto strings = p.get<std::vector<std::string>>("strings");
  VC_TEST_ASSERT((strings == std::vector<std::string>{"first", "second_value",
                                                      "third.value"}));

  const auto bools = p.get<std::vector<bool>>("bools");
  VC_TEST_ASSERT((bools == std::vector<bool>{true, false, true}));

  const auto single = p.get<std::vector<int>>("par4");
  VC_TEST_ASSERT((single == std::vector<int>{4}));

  for (const auto *key : {"invalid", "empty_item", "trailing_comma"}) {
    bool threw = false;
    try {
      (void)p.get<std::vector<int>>(key);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    VC_TEST_ASSERT(threw);
  }
}
