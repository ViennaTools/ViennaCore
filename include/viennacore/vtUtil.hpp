#pragma once

#include <iostream>

namespace vieTools {

// Small function to print a progress bar ()
inline void ProgressBar(size_t i, size_t finalCount = 100) {
  float progress = static_cast<float>(i) / static_cast<float>(finalCount);
  int barWidth = 70;

  std::cout << "[";
  int pos = static_cast<int>(static_cast<float>(barWidth) * progress);
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << static_cast<int>(progress * 100.0) << " %\r";
  std::cout.flush();
}

} // namespace vieTools
