#include <benchmark/benchmark.h>

#include <array>
#include <vector>

// Fixed arrays defining the size in bits and bit offset for each property.
static const int sizes[4] = {1, 1, 4, 2};
static const int offsets[4] = {0, 1, 2, 6};

// Data range: with these bit fields, the entire number is 1+1+4+2 = 8 bits.
constexpr int dataRange = 256;

// Benchmark approach 1: On-the-fly computing.
// For each data value, extract the properties using bit shifting and masking.
static void BM_OnTheFly(benchmark::State& state) {
  for (auto _ : state) {
    // Loop over all possible data values
    for (int data = 0; data < dataRange; ++data) {
      int values[4];
      for (int i = 0; i < 4; ++i) {
        // The expression extracts the bit-field according to the offset and
        // size.
        values[i] = (data >> offsets[i]) & ((1 << sizes[i]) - 1);
      }
      // Prevent the compiler from optimizing away the computation.
      benchmark::DoNotOptimize(values);
    }
  }
}
BENCHMARK(BM_OnTheFly);

// Utility function to precompute the lookup table.
// For every possible data value, compute and store all 4 extracted fields.
static std::vector<std::array<int, 4>> createLookupTable() {
  std::vector<std::array<int, 4>> table(dataRange);
  for (int data = 0; data < dataRange; ++data) {
    for (int i = 0; i < 4; ++i) {
      table[data][i] = (data >> offsets[i]) & ((1 << sizes[i]) - 1);
    }
  }
  return table;
}

// Benchmark approach 2: Using a precomputed lookup table.
// The lookup table is built once outside the timed loop.
static void BM_Precomputed(benchmark::State& state) {
  // Static ensures the table is computed only once.
  static const auto lookupTable = createLookupTable();
  for (auto _ : state) {
    for (int data = 0; data < dataRange; ++data) {
      // Lookup the precomputed values.
      const auto& values = lookupTable[data];
      benchmark::DoNotOptimize(values);
    }
  }
}
BENCHMARK(BM_Precomputed);

BENCHMARK_MAIN();
