#include <vcContext.hpp>
#include <vcLogger.hpp>
#include <vcTestAsserts.hpp>

int main() {
  using namespace viennacore;
  Logger::setLogLevel(LogLevel::DEBUG);

  Context context;

  context.create();
  context.destroy();
}