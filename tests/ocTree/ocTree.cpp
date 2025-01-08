#include <vcOcTree.hpp>

#include <iostream>

int main() {
  viennacore::OcTree tree(0, 0, 0, 10, 10, 10);

  tree.insert(1, 1, 1);
  tree.insert(1, 1, 2);
  tree.insert(1, 1, 5);

  std::cout << tree.find(1, 1, 1) << std::endl;

  return 0;
}