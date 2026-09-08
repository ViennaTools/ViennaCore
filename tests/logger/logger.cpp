#include <vcLogger.hpp>
#include <vcTestAsserts.hpp>

#include <sstream>

using namespace viennacore;

int main() {
  Logger &logger = Logger::getInstance();

  std::stringstream ss;

  const char *levelNames[][3] = {
      {"error", "ERROR", "Error"},
      {"warning", "WARNING", "WaRnInG"},
      {"info", "INFO", "Info"},
      {"intermediate", "INTERMEDIATE", "Intermediate"},
      {"timing", "TIMING", "TiMiNg"},
      {"debug", "DEBUG", "Debug"}};
  for (unsigned level = 0; level < 6; ++level) {
    for (const auto *name : levelNames[level]) {
      Logger::setLogLevel(name);
      VC_TEST_ASSERT(Logger::getLogLevel() == level);
    }
  }

  Logger::setLogLevel(std::string("iNfO"));
  VC_TEST_ASSERT(Logger::getLogLevel() == 2);
  for (const auto *name : {"unknown", ""}) {
    bool threw = false;
    try {
      Logger::setLogLevel(name);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    VC_TEST_ASSERT(threw);
    VC_TEST_ASSERT(Logger::getLogLevel() == 2);
  }

  logger.setLogLevel(LogLevel::TIMING);
  VC_TEST_ASSERT(logger.getLogLevel() == 4);

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

  VC_TEST_ASSERT(ss.str() == "    \033[1;33mWARNING: Warning message\n\033[0m");
  ss.str("");

  Logger::setLogFile("test_log.txt");
  VC_TEST_ASSERT(Logger::isLoggingToFile() == true);
  logger.addError("Error message", false);
  logger.print(std::cout);
}
