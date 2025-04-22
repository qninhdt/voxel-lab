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
  std::vector<std::tuple<int16_t, int16_t, int16_t>>
      target_values;  // Target triples for search

  // Different map implementations
  std::unordered_map<ivec3, int> coord_map;          // STL unordered_map
  tsl::robin_map<ivec3, int, ivec3_hash> robin_map;  // TSL robin_map
  tsl::robin_pg_map<ivec3, int, ivec3_hash>
      robin_flat_map;  // TSL flat robin map

  std::vector<int16_t> aligned_xs_avx2, aligned_ys_avx2, aligned_zs_avx2;
  std::vector<int16_t> aligned_xs_avx512, aligned_ys_avx512, aligned_zs_avx512;

  // Parameters for cluster generation
  struct ClusterParams {
    int16_t min_radius;           // Minimum radius of each cluster
    int16_t max_radius;           // Maximum radius of each cluster
    int16_t min_cluster_spacing;  // Minimum distance between cluster centers
    int16_t point_density;        // Density of points within a cluster (1-100%)
  };

  TestData(int num_clusters, int num_targets,
           const ClusterParams& params = {1, 4, 1000, 80}) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Distributions for random values
    std::uniform_int_distribution<int16_t> radius_dist(params.min_radius,
                                                       params.max_radius);
    std::uniform_int_distribution<int> density_dist(1, 100);

    // Create clusters with random positions ensuring minimum distance between
    // them
    std::vector<std::tuple<int16_t, int16_t, int16_t>> cluster_centers;
    std::vector<int16_t> cluster_radii;

    // Generate cluster centers
    int max_attempts = 100;  // Prevent infinite loops
    for (int c = 0; c < num_clusters; ++c) {
      bool valid_position = false;
      int attempts = 0;

      while (!valid_position && attempts < max_attempts) {
        // Generate a potential cluster center
        // Range large enough to fit all clusters with spacing
        int position_range = params.min_cluster_spacing *
                             static_cast<int>(sqrt(num_clusters) * 2);
        std::uniform_int_distribution<int16_t> pos_dist(-position_range,
                                                        position_range);

        int16_t cx = pos_dist(gen);
        int16_t cy = pos_dist(gen);
        int16_t cz = pos_dist(gen);

        // Check distance from all existing clusters
        valid_position = true;
        for (const auto& center : cluster_centers) {
          int16_t ex = std::get<0>(center);
          int16_t ey = std::get<1>(center);
          int16_t ez = std::get<2>(center);

          // Calculate squared distance
          int32_t dx = cx - ex;
          int32_t dy = cy - ey;
          int32_t dz = cz - ez;
          int32_t dist_squared = dx * dx + dy * dy + dz * dz;

          // Check if too close
          if (dist_squared <
              (params.min_cluster_spacing * params.min_cluster_spacing)) {
            valid_position = false;
            break;
          }
        }

        if (valid_position) {
          cluster_centers.emplace_back(cx, cy, cz);
          cluster_radii.push_back(radius_dist(gen));
        }

        attempts++;
      }

      if (attempts >= max_attempts) {
        // Couldn't place all clusters with desired spacing
        break;
      }
    }

    // Generate points for each cluster
    size_t total_points = 0;
    for (size_t c = 0; c < cluster_centers.size(); ++c) {
      int16_t center_x = std::get<0>(cluster_centers[c]);
      int16_t center_y = std::get<1>(cluster_centers[c]);
      int16_t center_z = std::get<2>(cluster_centers[c]);
      int16_t radius = cluster_radii[c];

      // Generate points within the cluster (a 3D grid with specified radius)
      for (int16_t dx = -radius; dx <= radius; ++dx) {
        for (int16_t dy = -radius; dy <= radius; ++dy) {
          for (int16_t dz = -radius; dz <= radius; ++dz) {
            // Only add point if it meets density requirement
            if (density_dist(gen) <= params.point_density) {
              int16_t px = static_cast<int16_t>(center_x + dx);
              int16_t py = static_cast<int16_t>(center_y + dy);
              int16_t pz = static_cast<int16_t>(center_z + dz);

              xs.push_back(px);
              ys.push_back(py);
              zs.push_back(pz);

              // Add to maps
              ivec3 coord = {px, py, pz};
              coord_map[coord] = static_cast<int>(total_points);
              robin_map[coord] = static_cast<int>(total_points);
              robin_flat_map[coord] = static_cast<int>(total_points);

              total_points++;
            }
          }
        }
      }
    }

    std::vector<size_t> indices(total_points);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    std::vector<int16_t> xs_copy(xs);
    std::vector<int16_t> ys_copy(ys);
    std::vector<int16_t> zs_copy(zs);
    for (size_t i = 0; i < total_points; ++i) {
      xs[i] = xs_copy[indices[i]];
      ys[i] = ys_copy[indices[i]];
      zs[i] = zs_copy[indices[i]];
    }

    // Resize data structures to match actual number of points
    size_t size = total_points;

    // Sorted copy (sorted lexicographically)
    std::vector<size_t> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
      if (xs[a] != xs[b]) return xs[a] < xs[b];
      if (ys[a] != ys[b]) return ys[a] < ys[b];
      return zs[a] < zs[b];
    });
    sorted_xs.resize(size);
    sorted_ys.resize(size);
    sorted_zs.resize(size);
    for (size_t i = 0; i < size; ++i) {
      sorted_xs[i] = xs[idx[i]];
      sorted_ys[i] = ys[idx[i]];
      sorted_zs[i] = zs[idx[i]];
    }

    // Aligned copies for SIMD (padded)
    // For int16_t, we can fit 16 elements in AVX2 (256-bit) and 32 in AVX-512
    // (512-bit)
    size_t avx2_pad = (size + 15) & ~15U, avx512_pad = (size + 31) & ~31U;
    aligned_xs_avx2.assign(avx2_pad, 0);
    aligned_ys_avx2.assign(avx2_pad, 0);
    aligned_zs_avx2.assign(avx2_pad, 0);
    aligned_xs_avx512.assign(avx512_pad, 0);
    aligned_ys_avx512.assign(avx512_pad, 0);
    aligned_zs_avx512.assign(avx512_pad, 0);
    for (size_t i = 0; i < size; ++i) {
      aligned_xs_avx2[i] = xs[i];
      aligned_ys_avx2[i] = ys[i];
      aligned_zs_avx2[i] = zs[i];
      aligned_xs_avx512[i] = xs[i];
      aligned_ys_avx512[i] = ys[i];
      aligned_zs_avx512[i] = zs[i];
    }

    // Create target values - some from actual data, some from non-existent
    // positions
    std::uniform_int_distribution<size_t> idxdist(0, size - 1);
    std::uniform_int_distribution<int> hit_miss_dist(0, 100);
    std::uniform_int_distribution<size_t> cluster_select(
        0, cluster_centers.size() - 1);

    for (size_t i = 0; i < num_targets; ++i) {
      // 80% hits, 20% misses for a realistic search pattern
      if (hit_miss_dist(gen) <= 80 && !xs.empty()) {
        // Hit - use an existing point
        size_t j = idxdist(gen);
        target_values.emplace_back(xs[j], ys[j], zs[j]);
      } else {
        // Miss - generate plausible but non-existent point
        if (!cluster_centers.empty()) {
          // Pick a random cluster
          size_t c = cluster_select(gen);
          int16_t center_x = std::get<0>(cluster_centers[c]);
          int16_t center_y = std::get<1>(cluster_centers[c]);
          int16_t center_z = std::get<2>(cluster_centers[c]);
          int16_t radius = cluster_radii[c];

          // Generate a point just outside the cluster radius
          std::uniform_int_distribution<int> dir_dist(
              0, 5);  // 6 directions (±x, ±y, ±z)
          int direction = dir_dist(gen);
          int16_t px = center_x, py = center_y, pz = center_z;

          switch (direction) {
            case 0:
              px = static_cast<int16_t>(center_x + radius + 1);
              break;
            case 1:
              px = static_cast<int16_t>(center_x - radius - 1);
              break;
            case 2:
              py = static_cast<int16_t>(center_y + radius + 1);
              break;
            case 3:
              py = static_cast<int16_t>(center_y - radius - 1);
              break;
            case 4:
              pz = static_cast<int16_t>(center_z + radius + 1);
              break;
            case 5:
              pz = static_cast<int16_t>(center_z - radius - 1);
              break;
          }

          target_values.emplace_back(px, py, pz);
        }
      }
    }

    // std::cout << "Generated " << size << " points in " << num_clusters
    //           << " clusters with " << num_targets << " targets.\n";
  }
};

std::map<size_t, std::shared_ptr<TestData>> test_data_map;
std::shared_ptr<TestData> getTestData(size_t size) {
  // std::cout << "Generating test data for size: " << size << "\n";
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

// === Benchmark: TSL Robin Flat Map lookup ===
static void RobinFlatMapLookup(benchmark::State& state) {
  size_t size = state.range(0);
  auto data = getTestData(size);
  const auto& robin_flat_map = data->robin_flat_map;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  for (auto _ : state) {
    auto [xt, yt, zt] = targets[tdist(gen)];
    ivec3 key = {xt, yt, zt};

    int index = -1;
    auto it = robin_flat_map.find(key);
    if (it != robin_flat_map.end()) {
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
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  for (auto _ : state) {
    int ii = tdist(gen);
    auto [xt, yt, zt] = targets[ii];
    int index = -1;
    for (size_t i = 0; i < size; ++i) {
      if (xs[i] == xt && ys[i] == yt && zs[i] == zt) {
        index = i;
        std::cout << "Found at index: " << index << "/" << xs.size() << "\n";
        break;
      }
    }
    benchmark::DoNotOptimize(index);
  }
  state.SetItemsProcessed(state.iterations() * size / 2);
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
    int left = 0, right = int(size) - 1, index = -1;
    while (left <= right) {
      int mid = left + (right - left) / 2;
      if (xs[mid] == xt && ys[mid] == yt && zs[mid] == zt) {
        index = mid;
        break;
      }
      if (std::tie(xs[mid], ys[mid], zs[mid]) < std::tie(xt, yt, zt))
        left = mid + 1;
      else
        right = mid - 1;
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
  const auto& xs = data->aligned_xs_avx2;
  const auto& ys = data->aligned_ys_avx2;
  const auto& zs = data->aligned_zs_avx2;
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
  const auto& xs = data->aligned_xs_avx512;
  const auto& ys = data->aligned_ys_avx512;
  const auto& zs = data->aligned_zs_avx512;
  const auto& targets = data->target_values;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, targets.size() - 1);

  for (auto _ : state) {
    auto [xt, yt, zt] = targets[tdist(gen)];
    int index = -1;
    __m512i vx = _mm512_set1_epi16(xt);
    __m512i vy = _mm512_set1_epi16(yt);
    __m512i vz = _mm512_set1_epi16(zt);
    size_t n = xs.size();
    for (size_t i = 0; i < n;
         i += 32) {  // Process 32 int16_t elements at a time
      __m512i vxi = _mm512_loadu_si512(reinterpret_cast<const void*>(&xs[i]));
      __m512i vyi = _mm512_loadu_si512(reinterpret_cast<const void*>(&ys[i]));
      __m512i vzi = _mm512_loadu_si512(reinterpret_cast<const void*>(&zs[i]));
      __mmask32 m1 = _mm512_cmpeq_epi16_mask(vxi, vx);
      __mmask32 m2 = _mm512_cmpeq_epi16_mask(vyi, vy);
      __mmask32 m3 = _mm512_cmpeq_epi16_mask(vzi, vz);
      __mmask32 allmask = m1 & m2 & m3;
      if (allmask) {
        unsigned int pos = __builtin_ctz(allmask);
        index = int(i) + pos;
        if (index < int(size)) break;
      }
    }
    benchmark::DoNotOptimize(index);
  }
  state.SetItemsProcessed(state.iterations() * size / 2);
#else
  state.SkipWithError("AVX-512 with BW not supported on this machine");
#endif
}

// === Register the benchmarks ===
// BENCHMARK(UnorderedMapLookup)
//     ->RangeMultiplier(2)
//     ->Range(1 << 1, 1 << 7)
//     ->Unit(benchmark::kMicrosecond);
// BENCHMARK(RobinMapLookup)
//     ->RangeMultiplier(2)
//     ->Range(1 << 1, 1 << 7)
//     ->Unit(benchmark::kMicrosecond);
// BENCHMARK(RobinFlatMapLookup)
//     ->RangeMultiplier(2)
//     ->Range(1 << 1, 1 << 7)
//     ->Unit(benchmark::kMicrosecond);
BENCHMARK(LinearSearch_Scalar)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 7)
    ->Unit(benchmark::kMicrosecond);
// BENCHMARK(BinarySearch_Scalar)
//     ->RangeMultiplier(2)
//     ->Range(1 << 1, 1 << 7)
//     ->Unit(benchmark::kMicrosecond);
BENCHMARK(LinearSearch_AVX2)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 7)
    ->Unit(benchmark::kMicrosecond);
// BENCHMARK(LinearSearch_AVX512)
//     ->RangeMultiplier(2)
//     ->Range(1 << 1, 1 << 7)
//     ->Unit(benchmark::kMicrosecond);
// BENCHMARK_MAIN();
