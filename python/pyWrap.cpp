/*
  This file is used to generate the python module of ViennaCS.
  It uses pybind11 to create the modules.
*/

#define PYBIND11_DETAILED_ERROR_MESSAGES
#define VIENNACORE_PYTHON_BUILD

// correct module name macro
#define TOKENPASTE_INTERNAL(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) TOKENPASTE_INTERNAL(x, y, z)
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define VIENNACS_MODULE_VERSION STRINGIZE(VIENNACS_VERSION)

#include <optional>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// all header files which define API functions
#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>
#include <vcTimer.hpp>

#include <omp.h>

using namespace viennacore;

PYBIND11_DECLARE_HOLDER_TYPE(Types, SmartPointer<Types>)

PYBIND11_MODULE(VIENNACS_MODULE_NAME, module) {
  module.doc() = "ViennaCore implements common functionality found in all "
                 "ViennaTools libraries.";

  // set version string of python module
  module.attr("__version__") = VIENNACS_MODULE_VERSION;

  // wrap omp_set_num_threads to control number of threads
  module.def("setNumThreads", &omp_set_num_threads);

  pybind11::enum_<LogLevel>(module, "LogLevel")
      .value("ERROR", LogLevel::ERROR)
      .value("WARNING", LogLevel::WARNING)
      .value("INFO", LogLevel::INFO)
      .value("TIMING", LogLevel::TIMING)
      .value("INTERMEDIATE", LogLevel::INTERMEDIATE)
      .value("DEBUG", LogLevel::DEBUG)
      .export_values();

  pybind11::class_<Logger, SmartPointer<Logger>>(module, "Logger")
      .def_static("setLogLevel", &Logger::setLogLevel)
      .def_static("getLogLevel", &Logger::getLogLevel)
      .def_static("getInstance", &Logger::getInstance,
                  pybind11::return_value_policy::reference)
      .def("addDebug", &Logger::addDebug)
      .def("addTiming",
           (Logger & (Logger::*)(std::string, double)) & Logger::addTiming)
      .def("addTiming", (Logger & (Logger::*)(std::string, double, double)) &
                            Logger::addTiming)
      .def("addInfo", &Logger::addInfo)
      .def("addWarning", &Logger::addWarning)
      .def("addError", &Logger::addError, pybind11::arg("s"),
           pybind11::arg("shouldAbort") = true)
      .def("print", [](Logger &instance) { instance.print(std::cout); });

  // Timer
  pybind11::class_<Timer<std::chrono::high_resolution_clock>>(module, "Timer")
      .def(pybind11::init<>())
      .def("start", &Timer<std::chrono::high_resolution_clock>::start,
           "Start the timer.")
      .def("finish", &Timer<std::chrono::high_resolution_clock>::finish,
           "Stop the timer.")
      .def("reset", &Timer<std::chrono::high_resolution_clock>::reset,
           "Reset the timer.")
      .def_readonly("currentDuration",
                    &Timer<std::chrono::high_resolution_clock>::currentDuration,
                    "Get the current duration of the timer in nanoseconds.")
      .def_readonly("totalDuration",
                    &Timer<std::chrono::high_resolution_clock>::totalDuration,
                    "Get the total duration of the timer in nanoseconds.");
}
