#pragma once

#include <vcLogger.hpp>

#include <fstream>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>

namespace viennacore {

namespace impl {
// Checks if a string starts with an - or not
inline bool isSigned(const std::string &s) {
  auto pos = s.find_first_not_of(' ');
  if (pos == std::string::npos)
    return false;
  if (s[pos] == '-')
    return true;
  return false;
}

// Converts string to the given numeric datatype
template <typename T> T convert(const std::string &s) {
  if constexpr (std::is_same_v<T, int>) {
    return std::stoi(s);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    if (isSigned(s))
      throw std::invalid_argument("The value must be unsigned");
    unsigned long int val = std::stoul(s);
    auto num = static_cast<unsigned int>(val);
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
  } else if constexpr (std::is_same_v<T, bool>) {
    if (s == "true")
      return true;
    if (s == "false")
      return false;
    throw std::invalid_argument("The value must be either 'true' or 'false'");
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
  } catch (std::exception &) {
    std::cout << '\'' << s << "' couldn't be converted to type  '"
              << typeid(value).name() << "'\n";
    return std::nullopt;
  }
  return {value};
}

// Class that can be used during the assigning process of a param map
// to the param struct
template <typename K, typename V, typename C = decltype(&convert<V>)>
class Item {
private:
  C conv;

public:
  K key;
  V &value;

  Item(K key_, V &value_) : conv(&convert<V>), key(key_), value(value_) {}

  Item(K key_, V &value_, C conv_) : conv(conv_), key(key_), value(value_) {}

  void operator()(const std::string &k) {
    try {
      value = conv(k);
    } catch (std::exception &) {
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
} // namespace impl

struct Parameters {
  std::unordered_map<std::string, std::string> m;

  bool exists(const std::string &key) const { return m.find(key) != m.end(); }

  template <typename T = double>
  [[nodiscard]] T get(const std::string &key) const {
    if (!exists(key)) {
      Logger::getInstance()
          .addError("Key not found in parameters: " + key)
          .print();
    }
    return impl::convert<T>(m.at(key));
  }

  // Opens a file and forwards its stream to the config stream parser.
  void readConfigFile(const std::string &filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
      std::cout << "Failed to open config file '" << filename << "'\n";
    }
    parseConfigStream(f);
  }

  void parseConfigStream(std::istream &input) {
    // Regex to find trailing and leading whitespaces
    const auto wsRegex = std::regex("^ +| +$|( ) +");

    // Regular expression for extracting key and value separated by '=' as two
    // separate capture groups
    const auto keyValueRegex = std::regex(
        R"rgx([ \t]*([0-9a-zA-Z_\-\.+]+)[ \t]*=[ \t]*([0-9a-zA-Z_\-\.+]+).*$)rgx");

    // Reads a simple config file containing a single <key>=<value> pair per
    // line and returns the content as an unordered map
    std::string line;
    while (std::getline(input, line)) {
      // Remove trailing and leading whitespaces
      line = std::regex_replace(line, wsRegex, "$1");
      // Skip this line if it is marked as a comment
      if (line.rfind('#') == 0 || line.empty())
        continue;

      // Extract key and value
      if (std::smatch smatch; std::regex_search(line, smatch, keyValueRegex)) {
        if (smatch.size() < 3) {
          Logger::getInstance().addWarning("Malformed line: " + line).print();
          continue;
        }

        m.insert({smatch[1], smatch[2]});
      }
    }
  }
};

} // namespace viennacore
