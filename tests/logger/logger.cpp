#include <vcLogger.hpp>
#include <vcTestAsserts.hpp>

#include <sstream>

using namespace viennacore;

int main() {
  Logger &logger = Logger::getInstance();

  std::stringstream ss;

  logger.setLogLevel(LogLevel::TIMING);
  VC_TEST_ASSERT(logger.getLogLevel() == 3);

  logger.setLogLevel(LogLevel::DEBUG);
  logger.addDebug("Debug message");
  logger.print(ss);

  VC_TEST_ASSERT(ss.str() == "    \033[1;32mDEBUG: Debug message\n\033[0m");
  ss.str("");

  logger.setLogLevel(LogLevel::TIMING);
  logger.addTiming("Timing message", 1.23);
  logger.print(ss);

  VC_TEST_ASSERT(ss.str().find("    Timing message: 1.23") == 0);
  ss.str("");

  logger.setLogLevel(LogLevel::INFO);
  logger.addInfo("Info message");
  logger.print(ss);

  VC_TEST_ASSERT(ss.str() == "    Info message\n\033[0m");
  ss.str("");

  logger.setLogLevel(LogLevel::WARNING);
  logger.addWarning("Warning message");
  logger.print(ss);

  VC_TEST_ASSERT(ss.str() ==
                 "\n    \033[1;33mWARNING: Warning message\n\033[0m");
  ss.str("");

  Logger::setLogFile("test_log.txt");
  VC_TEST_ASSERT(Logger::isLoggingToFile() == true);
  logger.addError("Error message", false);
  logger.print(std::cout);
}
