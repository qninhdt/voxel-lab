#include <benchmark/benchmark.h>
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
// Include TSL Robin Map
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
// Define a 3D vector structure for our keys
struct ivec3 {
  int16_t x, y, z;

  bool operator==(const ivec3& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct paded_ivec3 {
  union {
    struct {
      int16_t x, y, z;
    };
    long long v;
  };
};

// Define a hash function for ivec3 to use in hash maps
namespace std {

template <>
struct hash<ivec3> {
  size_t operator()(const ivec3& v) const {
    // Combine the hash of the components
    size_t h1 = std::hash<int16_t>()(v.x);
    size_t h2 = std::hash<int16_t>()(v.y);
    size_t h3 = std::hash<int16_t>()(v.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};
}  // namespace std

// Also provide hash for tsl::robin_map
struct ivec3_hash {
  size_t operator()(const ivec3& v) const {
    // Combine the hash of the components
    size_t h1 = std::hash<int16_t>()(v.x);
    size_t h2 = std::hash<int16_t>()(v.y);
    size_t h3 = std::hash<int16_t>()(v.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

// === Struct of arrays (SoA) test dataset with clustered data ===
struct TestData {
  std::vector<int16_t> xs, ys, zs;  // Original unsorted struct-of-arrays data
  std::vector<int16_t> sorted_xs, sorted_ys,
      sorted_zs;  // Sorted copy per coordinate
  std::vector<paded_ivec3> padded;
  std::vector<std::tuple<int16_t, int16_t, int16_t>>
      target_values;  // Target triples for search

  // Different map implementations
  std::unordered_map<ivec3, int> coord_map;  // STL unordered_map
  tsl::robin_map<ivec3, int> robin_map;      // TSL robin_map

  TestData(int size, int num_targets) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // generate point uniformly in 3D space, no clustering
    std::uniform_int_distribution<int16_t> dist(-10000, 10000);
    std::uniform_int_distribution<int16_t> radius(0, 0);

    std::unordered_set<ivec3> coord_set;
    for (size_t i = 0; i < size; ++i) {
      int16_t x, y, z;
      while (true) {
        x = dist(gen);
        y = dist(gen);
        z = dist(gen);
        if (coord_set.insert({x, y, z}).second)
          break;  // Insert and check for uniqueness
      }

      int r = radius(gen);

      // Generate points around the center
      for (int16_t dx = -r; dx <= r; ++dx) {
        for (int16_t dy = -r; dy <= r; ++dy) {
          for (int16_t dz = -r; dz <= r; ++dz) {
            xs.push_back(x + dx);
            ys.push_back(y + dy);
            zs.push_back(z + dz);
          }
        }
      }
    }

    // create padded vector
    padded.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
      padded.push_back({xs[i], ys[i], zs[i]});
    }

    // Create maps
    for (size_t i = 0; i < xs.size(); ++i) {
      ivec3 coord = {xs[i], ys[i], zs[i]};
      coord_map[coord] = static_cast<int>(i);
      robin_map[coord] = static_cast<int>(i);
    }
    // Sort the data for binary search
    std::vector<std::tuple<int16_t, int16_t, int16_t>> sorted_data;
    for (size_t i = 0; i < xs.size(); ++i) {
      sorted_data.emplace_back(xs[i], ys[i], zs[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end());
    for (const auto& [x, y, z] : sorted_data) {
      sorted_xs.push_back(x);
      sorted_ys.push_back(y);
      sorted_zs.push_back(z);
    }

    // create target values - some from actual data, some from non-existent
    std::uniform_int_distribution<size_t> idxdist(0, xs.size() - 1);
    std::uniform_int_distribution<int> hit_miss_dist(0, 100);
    // std::uniform_int_distribution<int16_t> (-10000, 10000);

    for (size_t i = 0; i < num_targets; ++i) {
      // 80% hits, 20% misses for a realistic search pattern
      if (hit_miss_dist(gen) <= 80 && !xs.empty()) {
        // Hit - use an existing point
        size_t j = idxdist(gen);
        target_values.emplace_back(xs[j], ys[j], zs[j]);
      } else {
        while (true) {
          // Miss - generate plausible but non-existent point
          int16_t x = dist(gen);
          int16_t y = dist(gen);
          int16_t z = dist(gen);
          if (coord_set.find({x, y, z}) == coord_set.end()) {
            target_values.emplace_back(x, y, z);
            break;
          }
        }
      }
    }

    std::cout << "Generated " << xs.size() << " points in " << size << "\n";
  }
};

std::map<size_t, std::shared_ptr<TestData>> test_data_map;
std::shared_ptr<TestData> getTestData(size_t size) {
  if (!test_data_map.count(size))
    test_data_map[size] = std::make_shared<TestData>(size, 1000);
  return test_data_map[size];
}

// === Benchmark: unordered_map lookup ===
static void UnorderedMapLookup(benchmark::State& state) {
  size_t size = state.range(0);
  auto data = getTestData(size);
  const auto& coord_map = data->coord_map;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  for (auto _ : state) {
    int ii = tdist(gen);
    auto [xt, yt, zt] = targets[ii];
    ivec3 key = {xt, yt, zt};
    int index = -1;
    auto it = coord_map.find(key);
    if (it != coord_map.end()) {
      index = it->second;
    }

    benchmark::DoNotOptimize(index);
  }
  // Each lookup is O(1) on average
  state.SetItemsProcessed(state.iterations());
}

// === Benchmark: TSL Robin Map lookup ===
static void RobinMapLookup(benchmark::State& state) {
  size_t size = state.range(0);
  auto data = getTestData(size);
  const auto& robin_map = data->robin_map;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  for (auto _ : state) {
    auto [xt, yt, zt] = targets[tdist(gen)];

    ivec3 key = {xt, yt, zt};

    int index = -1;
    auto it = robin_map.find(key);
    if (it != robin_map.end()) {
      index = it->second;
    }

    benchmark::DoNotOptimize(index);
  }
  state.SetItemsProcessed(state.iterations());
}

// === Benchmark: scalar linear search ===
static void LinearSearch_Scalar(benchmark::State& state) {
  size_t size = state.range(0);
  auto data = getTestData(size);
  const auto& xs = data->xs;
  const auto& ys = data->ys;
  const auto& zs = data->zs;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, xs.size() - 1);

  for (auto _ : state) {
    auto [xt, yt, zt] = targets[tdist(gen)];

    int index = -1;
    for (size_t i = 0; i < xs.size(); ++i) {
      if (xs[i] == xt && ys[i] == yt && zs[i] == zt) {
        index = i;
        break;
      }
    }
    benchmark::DoNotOptimize(index);
  }
  state.SetItemsProcessed(state.iterations() * size / 2);
}

inline size_t bit_floor(size_t i) {
  constexpr int num_bits = sizeof(i) * 8;
  return size_t(1) << (num_bits - std::countl_zero(i) - 1);
}
inline size_t bit_ceil(size_t i) {
  constexpr int num_bits = sizeof(i) * 8;
  return size_t(1) << (num_bits - std::countl_zero(i - 1));
}

template <typename It, typename T, typename Cmp>
It branchless_lower_bound(It begin, It end, const T& value, Cmp&& compare) {
  size_t length = end - begin;
  if (length == 0) return end;
  size_t step = bit_floor(length);
  if (step != length && compare(begin[step], value)) {
    length -= step + 1;
    if (length == 0) return end;
    step = bit_ceil(length);
    begin = end - step;
  }
  for (step /= 2; step != 0; step /= 2) {
    if (compare(begin[step], value)) begin += step;
  }
  return begin + compare(*begin, value);
}
template <typename It, typename T>
It branchless_lower_bound(It begin, It end, const T& value) {
  return branchless_lower_bound(begin, end, value, std::less<>{});
}

// === Benchmark: scalar binary search (lexicographical) ===
static void BinarySearch_Scalar(benchmark::State& state) {
  size_t size = state.range(0);
  auto data = getTestData(size);
  const auto& xs = data->sorted_xs;
  const auto& ys = data->sorted_ys;
  const auto& zs = data->sorted_zs;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  for (auto _ : state) {
    auto [xt, yt, zt] = targets[tdist(gen)];

    auto it = branchless_lower_bound(
        xs.begin(), xs.end(), xt, [&](const int16_t& a, const int16_t& b) {
          return std::tie(a, ys[&a - &xs[0]], zs[&a - &xs[0]]) <
                 std::tie(b, yt, zt);
        });
    int index = -1;
    if (it != xs.end() && *it == xt) {
      index = it - xs.begin();
      // std::cout << "Found at index: " << index << "/" << xs.size() << "\n";
    }

    benchmark::DoNotOptimize(index);
  }
  state.SetItemsProcessed(state.iterations() * std::ceil(std::log2(size)));
}

// === Benchmark: AVX2 vectorized linear search (SoA, 3-way match) ===
static void LinearSearch_AVX2(benchmark::State& state) {
#if defined(__AVX2__)
  size_t size = state.range(0);
  auto data = getTestData(size);
  const auto& xs = data->xs;
  const auto& ys = data->ys;
  const auto& zs = data->zs;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  for (auto _ : state) {
    auto [xt, yt, zt] = targets[tdist(gen)];
    int index = -1;
    __m256i vx = _mm256_set1_epi16(xt);
    __m256i vy = _mm256_set1_epi16(yt);
    __m256i vz = _mm256_set1_epi16(zt);
    size_t n = xs.size();
    for (size_t i = 0; i < n;
         i += 16) {  // Process 16 int16_t elements at a time
      __m256i vxi =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&xs[i]));
      __m256i vyi =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ys[i]));
      __m256i vzi =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&zs[i]));
      __m256i c1 = _mm256_cmpeq_epi16(vxi, vx);
      __m256i c2 = _mm256_cmpeq_epi16(vyi, vy);
      __m256i c3 = _mm256_cmpeq_epi16(vzi, vz);
      __m256i allmask = _mm256_and_si256(_mm256_and_si256(c1, c2), c3);
      int mask = _mm256_movemask_epi8(allmask);
      if (mask) {
        unsigned int pos = __builtin_ctz(mask) /
                           2;  // Divide by 2 since each int16_t is 2 bytes
        index = int(i) + pos;
        // std::cout << "Found at index: " << index << "/" << xs.size() << "\n";
        if (index < int(size)) break;
      }
    }
    benchmark::DoNotOptimize(index);
  }
  state.SetItemsProcessed(state.iterations() * size / 2);
#else
  state.SkipWithError("AVX2 not supported on this machine");
#endif
}

// === Benchmark: AVX-512 vectorized linear search (SoA, 3-way match) ===
static void LinearSearch_AVX512(benchmark::State& state) {
#if defined(__AVX512F__)
  size_t size = state.range(0);
  auto data = getTestData(size);
  const auto& xs = data->xs;
  const auto& ys = data->ys;
  const auto& zs = data->zs;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  const int16_t* xbase = xs.data();
  const int16_t* ybase = ys.data();
  const int16_t* zbase = zs.data();
  const int16_t* xend = xbase + xs.size();

  // [32, 32, 32, 0]
  union {
    size_t a[3];
    __m256i v;
  } vstep{32, 32, 32};

  for (auto _ : state) {
    auto [xt, yt, zt] = targets[tdist(gen)];

    __m512i vx = _mm512_set1_epi16(xt);
    __m512i vy = _mm512_set1_epi16(yt);
    __m512i vz = _mm512_set1_epi16(zt);

    // 3) Scan in 32‐element strides (32×2 B = 64 B per load, ×3 streams =
    // 192 B)
    // const int16_t* px = xbase;
    // const int16_t* py = ybase;
    // const int16_t* pz = zbase;
    union {
      struct {
        const int16_t* px;
        const int16_t* py;
        const int16_t* pz;
      };
      __m256i v;
    } p;
    p.px = xbase;
    p.py = ybase;
    p.pz = zbase;
    int index = -1;

    constexpr ptrdiff_t PREFETCH_DIST = 512;

    while (p.px < xend) {
      // software prefetch next chunk
      // _mm_prefetch((const char*)(p.px + PREFETCH_DIST / sizeof(*p.px)),
      //              _MM_HINT_T0);
      // _mm_prefetch((const char*)(p.py + PREFETCH_DIST / sizeof(*p.py)),
      //              _MM_HINT_T0);
      // _mm_prefetch((const char*)(p.pz + PREFETCH_DIST / sizeof(*p.pz)),
      //              _MM_HINT_T0);

      // aligned loads (assume your getTestData aligned these vectors to
      // 64 bytes)
      __m512i vx_i = _mm512_load_si512(p.px);
      __m512i vy_i = _mm512_load_si512(p.py);
      __m512i vz_i = _mm512_load_si512(p.pz);

      // compare & fuse masks
      __mmask32 m = _mm512_cmpeq_epi16_mask(vx_i, vx);
      m = _mm512_mask_cmpeq_epi16_mask(m, vy_i, vy);
      m = _mm512_mask_cmpeq_epi16_mask(m, vz_i, vz);

      if (m) {
        // find first matching lane (0..31)
        unsigned int lane = _tzcnt_u32(m);
        index = int((p.px - xbase) + lane);
        break;
      }

      p.v = _mm256_add_epi16(p.v, vstep.v);
    }
    benchmark::DoNotOptimize(index);
  }
  state.SetItemsProcessed(state.iterations() * size / 2);
#else
  state.SkipWithError("AVX-512 with BW not supported on this machine");
#endif
}

// === Benchmark: AVX-512 vectorized linear search (AoS, 3-way match) ===
static void LinearSearch_AVX512_AoS(benchmark::State& state) {
#if defined(__AVX512F__)
  size_t size = state.range(0);
  auto data = getTestData(size);
  const auto& padded = data->padded;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  for (auto _ : state) {
    // 1) pick target
    auto [xt, yt, zt] = targets[tdist(gen)];

    __m512i vT = _mm512_set1_epi64(paded_ivec3{xt, yt, zt}.v);

    int index = -1;
    const auto base = padded.data();
    const auto end = base + padded.size();

#define CHECK(offset)                                               \
  __m512i V##offset = _mm512_load_si512((__m512i*)(base + offset)); \
  __mmask8 m##offset = _mm512_cmpeq_epi64_mask(V##offset, vT);      \
  if (m##offset) {                                                  \
    unsigned lane = _tzcnt_u32(m##offset); /* 0..15 */              \
    index = int((ptr - base) + offset + lane);                      \
    break;                                                          \
  }

    for (auto ptr = base; ptr + 64 <= end; ptr += 64) {
      CHECK(0);
      CHECK(8);
      CHECK(16);
      CHECK(24);
      CHECK(32);
      CHECK(40);
      CHECK(48);
      CHECK(56);
    }

    benchmark::DoNotOptimize(index);
  }

  state.SetItemsProcessed(state.iterations() * size / 2);
#else
  state.SkipWithError("AVX-512 with BW not supported on this machine");
#endif
}
// === Register the benchmarks ===
const int kMinSize = 1 << 0;  // 1
const int kMaxSize = 1 << 8;  // 256

BENCHMARK(UnorderedMapLookup)
    ->RangeMultiplier(4)
    ->Range(kMinSize, kMaxSize)
    ->Unit(benchmark::kMicrosecond);

// // BENCHMARK(RobinMapLookup)
// //     ->RangeMultiplier(4)
// //     ->Range(kMinSize, kMaxSize)
// //     ->Unit(benchmark::kMicrosecond);

// BENCHMARK(LinearSearch_Scalar)
//     ->RangeMultiplier(4)
//     ->Range(kMinSize, kMaxSize)
//     ->Unit(benchmark::kMicrosecond);

// BENCHMARK(BinarySearch_Scalar)
//     ->RangeMultiplier(4)
//     ->Range(kMinSize, kMaxSize)
//     ->Unit(benchmark::kMicrosecond);

// BENCHMARK(LinearSearch_AVX2)
//     ->RangeMultiplier(4)
//     ->Range(kMinSize, kMaxSize)
//     ->Unit(benchmark::kMicrosecond);

BENCHMARK(LinearSearch_AVX512)
    ->RangeMultiplier(4)
    ->Range(kMinSize, kMaxSize)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(LinearSearch_AVX512_AoS)
    ->RangeMultiplier(4)
    ->Range(kMinSize, kMaxSize)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
