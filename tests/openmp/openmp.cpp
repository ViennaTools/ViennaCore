#include <omp.h>

int main() {
  int sum = 0;
#pragma omp parallel
  {
#pragma omp atomic
    sum += 1;
  }
  return sum;
}