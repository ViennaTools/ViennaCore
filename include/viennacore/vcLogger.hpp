#pragma once

#include "vcTimer.hpp"

#include <iostream>
#include <string>

#ifdef VIENNACORE_COMPILE_GPU
#include <cuda.h>
#endif

#define TM_RED "\033[1;31m"
#define TM_GREEN "\033[1;32m"
#define TM_YELLOW "\033[1;33m"
#define TM_BLUE "\033[1;34m"
#define TM_RESET "\033[0m"
#define TM_DEFAULT TM_RESET
#define TM_BOLD "\033[1;1m"

namespace viennacore {
  // verbosity levels:
  // 0 errors
  // 1 + warnings
  // 2 + info
  // 3 + timings
  // 4 + intermediate output (meshes)
  // 5 + debug
  enum class LogLevel : unsigned {
    ERROR = 0,
    WARNING = 1,
    INFO = 2,
    TIMING = 3,
    INTERMEDIATE = 4,
    DEBUG = 5
  };

  /// Singleton class for thread-safe logging. The logger can be accessed via
  /// Logger::getInstance(). The logger can be configured to print messages of a
  /// certain level or lower. The default level is INFO. The different logging
  /// levels are: ERROR, WARNING, INFO, TIMING, INTERMEDIATE, DEBUG. The logger
  /// can also be used to print timing information.
  class Logger {
    std::string message;

    bool error = false;
    const unsigned tabWidth = 4;
    static LogLevel logLevel;

    Logger() {
    }

  public:
    // delete constructors to result in better error messages by compilers
    Logger(const Logger &) = delete;

    void operator=(const Logger &) = delete;

    // Set the log level for all instances of the logger.
    static void setLogLevel(const LogLevel passedLogLevel) {
      logLevel = passedLogLevel;
    }

    static unsigned getLogLevel() { return static_cast<unsigned>(logLevel); }

    static Logger &getInstance() {
      static Logger instance;
      return instance;
    }

    // Add debug message if log level is high enough.
    Logger &addDebug(const std::string &s) {
      if (getLogLevel() < 5)
        return *this;
#pragma omp critical
      {
        message += std::string(tabWidth, ' ') + TM_GREEN + "DEBUG: " + s + "\n";
      }
      return *this;
    }

    // Add timing message if log level is high enough.
    template<class Clock>
    Logger &addTiming(const std::string &s, Timer<Clock> &timer) {
      if (getLogLevel() < 3)
        return *this;
#pragma omp critical
      {
        message += std::string(tabWidth, ' ') + s +
            " took: " + std::to_string(timer.currentDuration * 1e-9) +
            " s \n";
      }
      return *this;
    }

    Logger &addTiming(const std::string &s, double timeInSeconds) {
      if (getLogLevel() < 3)
        return *this;
#pragma omp critical
      {
        message += std::string(tabWidth, ' ') + s + ": " +
            std::to_string(timeInSeconds) + " s \n";
      }
      return *this;
    }

    Logger &addTiming(const std::string &s, double timeInSeconds,
                      double totalTimeInSeconds) {
      if (getLogLevel() < 3)
        return *this;
#pragma omp critical
      {
        message += std::string(tabWidth, ' ') + s + ": " +
            std::to_string(timeInSeconds) + " s\n" +
            std::string(tabWidth, ' ') + "Percent of total time: " +
            std::to_string(timeInSeconds / totalTimeInSeconds * 100) +
            "\n";
      }
      return *this;
    }

    // Add info message if log level is high enough.
    Logger &addInfo(const std::string &s) {
      if (getLogLevel() < 2)
        return *this;
#pragma omp critical
      {
        message += std::string(tabWidth, ' ') + s + "\n";
      }
      return *this;
    }

    // Add warning message if log level is high enough.
    Logger &addWarning(const std::string &s) {
      if (getLogLevel() < 1)
        return *this;
#pragma omp critical
      {
        message += "\n" + std::string(tabWidth, ' ') + TM_YELLOW +
            "WARNING: " + s + "\n";
      }
      return *this;
    }

    // Add error message if log level is high enough.
    Logger &addError(const std::string &s, const bool shouldAbort = true) {
#pragma omp critical
      {
        message +=
            "\n" + std::string(tabWidth, ' ') + TM_RED + "ERROR: " + s + "\n";
        // always abort once error message should be printed
        error = true;
      }
      // abort now if asked
      if (shouldAbort)
        print();
      return *this;
    }

#ifdef VIENNACORE_COMPILE_GPU
  std::string getErrorString(CUresult err) {
    const char *errorMsg[2048];
    cuGetErrorString(err, errorMsg);
    std::string errorString = *errorMsg;
    return errorString;
  }

  Logger &addModuleError(std::string moduleName, CUresult err,
                         bool shouldAbort = true) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + TM_RED +
                 "ERROR in CUDA module " + moduleName + ": " +
                 getErrorString(err) + "\n";
      // always abort once error message should be printed
      error = true;
    }
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  Logger &addFunctionError(const std::string &kernelName, CUresult err,
                           const bool shouldAbort = true) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + TM_RED +
                 "ERROR in CUDA kernel " + kernelName + ": " +
                 getErrorString(err) + "\n";
      // always abort once error message should be printed
      error = true;
    }
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  Logger &addLaunchError(const std::string &kernelName, CUresult err,
                         const bool shouldAbort = true) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + TM_RED +
                 "ERROR in CUDA kernel launch (" + kernelName +
                 "): " + getErrorString(err) + "\n";
      // always abort once error message should be printed
      error = true;
    }
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }
#endif

    // Print message to std::cout if log level is high enough.
    void print(std::ostream &out = std::cout) {
#pragma omp critical
      {
        out << message;
        out << TM_RESET;
        message.clear();
        out.flush();
        if (error)
          abort();
      }
    }
  };

  // initialize static member of logger
  inline LogLevel Logger::logLevel = LogLevel::INFO;
} // namespace viennacore
