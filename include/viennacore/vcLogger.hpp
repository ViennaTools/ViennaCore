#pragma once

#include "vcTimer.hpp"

#include <fstream>
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
// 3 + intermediate output (points)
// 4 + timings
// 5 + debug
enum class LogLevel : unsigned {
  ERROR = 0,
  WARNING = 1,
  INFO = 2,
  INTERMEDIATE = 3,
  TIMING = 4,
  DEBUG = 5
};

/// Singleton class for thread-safe logging. The logger can be accessed via
/// Logger::getInstance(). The logger can be configured to print messages of a
/// certain level or lower. The default level is INFO. The different logging
/// levels are: ERROR, WARNING, INFO, TIMING, INTERMEDIATE, DEBUG. The logger
/// can also be used to print timing information.
class Logger {
  std::string message;
  static std::ofstream logFile;
  static bool logToFile;

  bool error = false;
  std::string color = "";
  const std::string tab = std::string(4, ' ');
  static LogLevel logLevel;

  Logger() {}

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

  bool hasError() const { return error; }

  // Enable logging to a file. Creates or overwrites the file.
  static bool setLogFile(const std::string &filename) {
    if (logFile.is_open()) {
      logFile.close();
    }
    logFile.open(filename, std::ios::out | std::ios::trunc);
    logToFile = logFile.is_open();
    if (!logToFile) {
      getInstance().addError("Failed to open log file: " + filename, false);
    }
    return logToFile;
  }

  // Append to existing log file or create new one
  static bool appendToLogFile(const std::string &filename) {
    if (logFile.is_open()) {
      logFile.close();
    }
    logFile.open(filename, std::ios::out | std::ios::app);
    logToFile = logFile.is_open();
    if (!logToFile) {
      getInstance().addError(
          "Failed to open log file for appending: " + filename, false);
    }
    return logToFile;
  }

  // Disable file logging and close the log file
  static void closeLogFile() {
    if (logFile.is_open()) {
      logFile.close();
    }
    logToFile = false;
  }

  // Check if file logging is enabled
  static bool isLoggingToFile() { return logToFile; }

  // Add debug message if log level is high enough.
  Logger &addDebug(const std::string &s) {
    if (getLogLevel() < 5)
      return *this;
#pragma omp critical
    {
      message += "DEBUG: " + s + "\n";
      color = TM_GREEN; // Set color for debug messages
    }
    return *this;
  }

  // Add timing message if log level is high enough.
  template <class Clock>
  Logger &addTiming(const std::string &s, Timer<Clock> &timer) {
    if (getLogLevel() < 4)
      return *this;
#pragma omp critical
    {
      message += s + " took: " + std::to_string(timer.currentDuration * 1e-9) +
                 " s \n";
    }
    return *this;
  }

  Logger &addTiming(const std::string &s, double timeInSeconds) {
    if (getLogLevel() < 4)
      return *this;
#pragma omp critical
    { message += s + ": " + std::to_string(timeInSeconds) + " s \n"; }
    return *this;
  }

  Logger &addTiming(const std::string &s, double timeInSeconds,
                    double totalTimeInSeconds) {
    if (getLogLevel() < 4)
      return *this;
#pragma omp critical
    {
      message += s + ": " + std::to_string(timeInSeconds) + " s\n" + tab +
                 "Percent of total time: " +
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
    { message += s + "\n"; }
    return *this;
  }

  // Add warning message if log level is high enough.
  Logger &addWarning(const std::string &s) {
    if (getLogLevel() < 1)
      return *this;
#pragma omp critical
    {
      message += "WARNING: " + s + "\n";
      color = TM_YELLOW; // Set color for warning messages
    }
    return *this;
  }

  // Add error message if log level is high enough.
  Logger &addError(const std::string &s, const bool shouldAbort = true) {
#pragma omp critical
    {
      message += (shouldAbort ? "" : "ERROR: ") + s + (shouldAbort ? "" : "\n");
      error = shouldAbort;
      color = TM_RED; // Set color for error messages
    }
    return *this;
  }

#ifdef VIENNACORE_COMPILE_GPU
  std::string getErrorString(CUresult err) {
    const char *errorMsg[2048];
    cuGetErrorString(err, errorMsg);
    std::string errorString = *errorMsg;
    return errorString;
  }

  Logger &addModuleError(const std::string &moduleName, CUresult err) {
#pragma omp critical
    {
      message += "ERROR in CUDA module " + moduleName + ": " +
                 getErrorString(err) + "\n";
      error = true;   // always abort once error message should be printed
      color = TM_RED; // Set color for error messages
    }
    return *this;
  }

  Logger &addFunctionError(const std::string &kernelName, CUresult err) {
#pragma omp critical
    {
      message += "ERROR in CUDA kernel " + kernelName + ": " +
                 getErrorString(err) + "\n";
      error = true;   // always abort once error message should be printed
      color = TM_RED; // Set color for error messages
    }
    return *this;
  }

  Logger &addLaunchError(const std::string &kernelName, CUresult err) {
#pragma omp critical
    {
      message += "ERROR in CUDA kernel launch (" + kernelName +
                 "): " + getErrorString(err) + "\n";
      error = true;   // always abort once error message should be printed
      color = TM_RED; // Set color for error messages
    }
    return *this;
  }
#endif

  // Print message to std::cout if log level is high enough.
  void print(std::ostream &out = std::cout) {
    if (message.empty())
      return; // nothing to print

    std::string errorMsg;

#pragma omp critical
    {
      if (!error) {
        out << tab << color << message << TM_RESET;
        color.clear(); // Reset color for next messages
      } else {
        errorMsg = message;
      }

      // Also write to file if file logging is enabled
      if (logToFile && logFile.is_open()) {
        logFile << message;
        logFile.flush();
      }

      message.clear();
      out.flush();
    }

    if (error) {
      error = false; // reset error state
      throw std::runtime_error(errorMsg);
    }
  }
};

// initialize static members of logger

#ifndef NDEBUG
inline LogLevel Logger::logLevel = LogLevel::DEBUG;
#else
inline LogLevel Logger::logLevel = LogLevel::INFO;
#endif
inline std::ofstream Logger::logFile;
inline bool Logger::logToFile = false;
} // namespace viennacore
