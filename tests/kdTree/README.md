# KD-tree benchmark

Build with optimizations enabled:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DVIENNACORE_BUILD_TESTS=ON
cmake --build build-release --target KDTreeBenchmark -j
OMP_NUM_THREADS=4 ./build-release/tests/kdTree/KDTreeBenchmark
```

The default benchmark compares KDTree and NFKDTree with 1,000, 10,000,
100,000, and 1,000,000 points. Each implementation gets exactly 10 runs per
point count. Build and lookup times are reported as independent minima in
milliseconds; lookup time covers the complete batch of queries.

To select the query count, ordinary radius, and point counts:

```sh
./build-release/tests/kdTree/KDTreeBenchmark --queries 2000 --radius 0.02 1000 10000 100000
```

Both trees use identical seeded double-precision 3D points and queries sampled
uniformly from the unit cube. Each run constructs a fresh tree. Build timing
covers `build()` only, excluding input generation, copying, and destruction.
Lookup timing includes serial `findNearestWithinRadius()` calls, allocation,
result consumption, and result destruction. Both searches receive a squared
radius, with NFKDTree's threshold adjusted to include the boundary like KDTree.
KDTree sorts its results internally; NFKDTree returns them unsorted.
Non-error logging is disabled so empty-result warnings do not affect timings.

Backend order alternates between runs. Match counts and index sums are compared
after timing, and a mismatch fails the benchmark. Output includes CSV columns
after the lines beginning with `#`. Set `OMP_NUM_THREADS` to control build
parallelism; queries run serially. With OpenMP disabled, builds are serial too.

## Saved results and plots

`results/kdtree_benchmark.csv` contains a Release run using four build threads,
with the run date, CPU, compiler, and command recorded in comment lines. The
adjacent PNG and SVG show build time and total radius-query time on logarithmic
axes. Timings are specific to the recorded machine and workload.

Regenerate both plots using Python with matplotlib installed:

```sh
python tests/kdTree/plot_benchmark.py
```

The script defaults to the saved CSV and writes PNG and SVG files beside it.
To plot a different run or choose another output location:

```sh
python tests/kdTree/plot_benchmark.py path/to/results.csv --output-prefix path/to/timings
```
