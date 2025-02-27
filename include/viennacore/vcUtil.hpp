#pragma once

#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdio.h>
#include <string>
#include <type_traits>
#include <unordered_map>

#ifdef __WIN32__
#define vc_snprintf sprintf_s
#else
#define vc_snprintf snprintf
#endif

namespace viennacore {

namespace util {

// Checks if a string starts with a - or not
[[nodiscard]] inline bool isSigned(const std::string &s) {
  auto pos = s.find_first_not_of(' ');
  if (pos == s.npos)
    return false;
  if (s[pos] == '-')
    return true;
  return false;
}

// Converts string to the given numeric datatype
template <typename T> [[nodiscard]] T convert(const std::string &s) {
  if constexpr (std::is_same_v<T, int>) {
    return std::stoi(s);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    if (isSigned(s))
      throw std::invalid_argument("The value must be unsigned");
    unsigned long int val = std::stoul(s);
    unsigned int num = static_cast<unsigned int>(val);
    if (val != num)
      throw std::out_of_range("The value is larger than the largest value "
                              "representable by `unsigned int`.");
    return num;
  } else if constexpr (std::is_same_v<T, long int>) {
    return std::stol(s);
  } else if constexpr (std::is_same_v<T, unsigned long int>) {
    if (isSigned(s))
      throw std::invalid_argument("The value must be unsigned");
    return std::stoul(s);
  } else if constexpr (std::is_same_v<T, long long int>) {
    return std::stoll(s);
  } else if constexpr (std::is_same_v<T, unsigned long long int>) {
    if (isSigned(s))
      throw std::invalid_argument("The value must be unsigned");
    return std::stoull(s);
  } else if constexpr (std::is_same_v<T, float>) {
    return std::stof(s);
  } else if constexpr (std::is_same_v<T, double>) {
    return std::stod(s);
  } else if constexpr (std::is_same_v<T, long double>) {
    return std::stold(s);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return s;
  } else {
    // Throws a compile time error for all types but void
    return;
  }
}

// safeConvert wraps the convert function to catch exceptions. If an error
// occurs the default initialized value is returned.
template <typename T> std::optional<T> safeConvert(const std::string &s) {
  T value;
  try {
    value = convert<T>(s);
  } catch (std::exception &e) {
    std::cout << '\'' << s << "' couldn't be converted to type  '"
              << typeid(value).name() << "'\n";
    return std::nullopt;
  }
  return {value};
}

inline std::unordered_map<std::string, std::string>
parseConfigStream(std::istream &input) {
  // Regex to find trailing and leading whitespaces
  const auto wsRegex = std::regex("^ +| +$|( ) +");

  // Regular expression for extracting key and value separated by '=' as two
  // separate capture groups
  const auto keyValueRegex = std::regex(
      R"rgx([ \t]*([0-9a-zA-Z_\-\.+]+)[ \t]*=[ \t]*([0-9a-zA-Z_\-\.+]+).*$)rgx");

  // Reads a simple config file containing a single <key>=<value> pair per line
  // and returns the content as an unordered map
  std::unordered_map<std::string, std::string> paramMap;
  std::string line;
  while (std::getline(input, line)) {
    // Remove trailing and leading whitespaces
    line = std::regex_replace(line, wsRegex, "$1");
    // Skip this line if it is marked as a comment
    if (line.rfind('#') == 0 || line.empty())
      continue;

    // Extract key and value
    std::smatch smatch;
    if (std::regex_search(line, smatch, keyValueRegex)) {
      if (smatch.size() < 3) {
        std::cerr << "Malformed line:\n " << line;
        continue;
      }

      paramMap.insert({smatch[1], smatch[2]});
    }
  }
  return paramMap;
}

// Opens a file and forwards its stream to the config stream parser.
inline std::unordered_map<std::string, std::string>
readFile(const std::string &filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cout << "Failed to open config file '" << filename << "'\n";
    return {};
  }
  return parseConfigStream(f);
}

// Class that can be used during the assigning process of a param map to the
// param struct
template <typename K, typename V, typename C = decltype(&convert<V>)>
class Item {
private:
  C conv;

public:
  K key;
  V &value;

  Item(K key_, V &value_) : key(key_), value(value_), conv(&convert<V>) {}

  Item(K key_, V &value_, C conv_) : key(key_), value(value_), conv(conv_) {}

  void operator()(const std::string &k) {
    try {
      value = conv(k);
    } catch (std::exception &e) {
      std::cout << '\'' << k << "' couldn't be converted to type of parameter '"
                << key << "'\n";
    }
  }
};

// If the key is found in the unordered_map, then the
template <typename K, typename V, typename C>
void AssignItems(std::unordered_map<std::string, std::string> &map,
                 Item<K, V, C> &&item) {
  if (auto it = map.find(item.key); it != map.end()) {
    item(it->second);
    // Remove the item from the map, since it is now 'consumed'.
    map.erase(it);
  } else {
    std::cout << "Couldn't find '" << item.key
              << "' in parameter file. Using default value instead.\n";
  }
}

// Peels off items from parameter pack
template <typename K, typename V, typename C, typename... ARGS>
void AssignItems(std::unordered_map<std::string, std::string> &map,
                 Item<K, V, C> &&item, ARGS &&...args) {
  AssignItems(map, std::forward<Item<K, V, C>>(item));
  AssignItems(map, std::forward<ARGS>(args)...);
}

template <class NumericType, std::size_t D>
std::string arrayToString(const std::array<NumericType, D> arr) {
  std::stringstream arrayStr;
  arrayStr << "[";
  for (std::size_t i = 0; i < D - 1; i++) {
    arrayStr << arr[i] << ", ";
  }
  arrayStr << arr[D - 1] << "]";
  return arrayStr.str();
}

struct Parameters {
  std::unordered_map<std::string, std::string> m;

  void readConfigFile(const std::string &fileName) { m = readFile(fileName); }

  template <typename T = double>
  [[nodiscard]] T get(const std::string &key) const {
    if (m.find(key) == m.end()) {
      std::cout << "Key not found in parameters: " << key << std::endl;
      exit(1);
      return T();
    }
    return convert<T>(m.at(key));
  }
};

// Small function to print a progress bar ()
inline void ProgressBar(size_t i, size_t finalCount = 100) {
  float progress = static_cast<float>(i) / static_cast<float>(finalCount);
  int barWidth = 70;

  std::cout << "[";
  int pos = static_cast<int>(static_cast<float>(barWidth) * progress);
  for (int j = 0; j < barWidth; ++j) {
    if (j < pos)
      std::cout << "=";
    else if (j == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << static_cast<int>(progress * 100.0) << " %\r";
  std::cout.flush();
}

// Print function for large numbers, printing 10000000 as "10M" instead
inline std::string prettyDouble(const double val) {
  const double absVal = std::abs(val);
  char result[1000];

  if (absVal >= 1e+18f)
    vc_snprintf(result, 1000, "%.1f%c", val / 1e18f, 'E');
  else if (absVal >= 1e+15f)
    vc_snprintf(result, 1000, "%.1f%c", val / 1e15f, 'P');
  else if (absVal >= 1e+12f)
    vc_snprintf(result, 1000, "%.1f%c", val / 1e12f, 'T');
  else if (absVal >= 1e+09f)
    vc_snprintf(result, 1000, "%.1f%c", val / 1e09f, 'G');
  else if (absVal >= 1e+06f)
    vc_snprintf(result, 1000, "%.1f%c", val / 1e06f, 'M');
  else if (absVal >= 1e+03f)
    vc_snprintf(result, 1000, "%.1f%c", val / 1e03f, 'k');
  else if (absVal <= 1e-12f)
    vc_snprintf(result, 1000, "%.1f%c", val * 1e15f, 'f');
  else if (absVal <= 1e-09f)
    vc_snprintf(result, 1000, "%.1f%c", val * 1e12f, 'p');
  else if (absVal <= 1e-06f)
    vc_snprintf(result, 1000, "%.1f%c", val * 1e09f, 'n');
  else if (absVal <= 1e-03f)
    vc_snprintf(result, 1000, "%.1f%c", val * 1e06f, 'u');
  else if (absVal <= 1e-00f)
    vc_snprintf(result, 1000, "%.1f%c", val * 1e03f, 'm');
  else
    vc_snprintf(result, 1000, "%f", (float)val);

  return result;
}

} // namespace util

} // namespace viennacore
