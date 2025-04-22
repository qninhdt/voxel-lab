#include <benchmark/benchmark.h>

#include <iostream>
#include <random>
#include <vector>

// runtime bit_buffer
// --------------------------------------------------------------
class runtime_bit_buffer {
 public:
  runtime_bit_buffer(size_t size, size_t bit_size) {
    // Allocate buffer
    SIZE = size;
    BIT_SIZE = bit_size;

    const uint32_t BITS_PER_VALUE = 1 << BIT_SIZE;
    const uint32_t TOTAL_INTS = (SIZE * BITS_PER_VALUE + 31) >> 5;

    FIVE_MINUS_BIT_SIZE = 5 - BIT_SIZE;
    VALUES_PER_INT_MINUS_ONE = (32 >> BIT_SIZE) - 1;
    BIT_MASK = 0xFFFFFFFF >> (32 - BITS_PER_VALUE);

    MAX_VALUE = 0xFFFFFFFF >> (32 - BITS_PER_VALUE);

    // Allocate buffer
    buffer_ = new uint32_t[TOTAL_INTS];
    std::fill(buffer_, buffer_ + TOTAL_INTS, 0);
  }

  uint32_t get(const uint32_t& index) const {
    assert(index < SIZE);

    uint32_t intIndex = index >> FIVE_MINUS_BIT_SIZE;  // index / 32
    uint32_t bitOffset = (index & VALUES_PER_INT_MINUS_ONE)
                         << BIT_SIZE;  // index % 32

    return (buffer_[intIndex] >> bitOffset) & BIT_MASK;
  }

  void set(const uint32_t& index, const uint32_t& value) {
    assert(index < SIZE);
    assert(value <= MAX_VALUE);

    // index / 32
    uint32_t word_idx = index >> FIVE_MINUS_BIT_SIZE;

    // index % 32
    uint32_t offset = (index & VALUES_PER_INT_MINUS_ONE) << BIT_SIZE;

    uint32_t word = buffer_[word_idx];
    uint32_t mask = BIT_MASK << offset;

    word = ((word & ~mask) | (value << offset));
    buffer_[word_idx] = word;
  }

  ~runtime_bit_buffer() { delete[] buffer_; }

 private:
  uint32_t* buffer_;                 // 8 bytes
  uint32_t BIT_MASK;                 // 4 bytes
  uint8_t FIVE_MINUS_BIT_SIZE;       // 1 byte
  uint8_t BIT_SIZE;                  // 1 byte
  uint8_t VALUES_PER_INT_MINUS_ONE;  // 1 byte

  uint32_t MAX_VALUE;
  uint32_t SIZE;
};

// complied time bit_buffer
// -------------------------------------------------------------

class compiled_time_bit_buffer {
 public:
  compiled_time_bit_buffer() {}

  virtual void set(const uint32_t& index, const uint32_t& value) = 0;
  virtual uint32_t get(const uint32_t& index) const = 0;
};
template <size_t SIZE, size_t BIT_SIZE>
class compiled_time_bit_buffer_impl : public compiled_time_bit_buffer {
 public:
  static_assert(SIZE > 0, "Size must be greater than 0");
  static_assert(BIT_SIZE >= 0 && BIT_SIZE <= 5,
                "Bit size must be between 1 and 5");

  static constexpr size_t BITS_PER_VALUE = 1 << BIT_SIZE;
  static constexpr size_t FIVE_MINUS_BIT_SIZE = 5 - BIT_SIZE;
  static constexpr size_t VALUES_PER_INT = 32 >> BIT_SIZE;
  static constexpr size_t VALUES_PER_INT_MINUS_ONE = VALUES_PER_INT - 1;
  static constexpr size_t MAX_VALUE = (0xFFFFFFFF >> (32 - BITS_PER_VALUE));
  static constexpr size_t TOTAL_INTS = (SIZE * BITS_PER_VALUE + 31) >> 5;
  static constexpr size_t BIT_MASK = (0xFFFFFFFF >> (32 - BITS_PER_VALUE));

  void set(const uint32_t& index, const uint32_t& value) override {
    assert(index < SIZE);
    assert(value <= MAX_VALUE);

    uint32_t intIndex = index >> FIVE_MINUS_BIT_SIZE;
    uint32_t bitOffset = (index & VALUES_PER_INT_MINUS_ONE) << BIT_SIZE;
    uint32_t clearMask = ~(BIT_MASK << bitOffset);

    buffer_[intIndex] = (buffer_[intIndex] & clearMask) | (value << bitOffset);
  }

  uint32_t get(const uint32_t& index) const override {
    assert(index < SIZE);

    uint32_t intIndex = index >> FIVE_MINUS_BIT_SIZE;
    uint32_t bitOffset = (index & VALUES_PER_INT_MINUS_ONE) << BIT_SIZE;

    return (buffer_[intIndex] >> bitOffset) & BIT_MASK;
  }

 private:
  uint32_t buffer_[TOTAL_INTS] = {0};
};

template <size_t SIZE>
static compiled_time_bit_buffer* create_compiled_time_bit_buffer(
    uint32_t bit_size) {
  switch (bit_size) {
    case 0:
      return new compiled_time_bit_buffer_impl<SIZE, 0>();
    case 1:
      return new compiled_time_bit_buffer_impl<SIZE, 1>();
    case 2:
      return new compiled_time_bit_buffer_impl<SIZE, 2>();
    case 3:
      return new compiled_time_bit_buffer_impl<SIZE, 3>();
    case 4:
      return new compiled_time_bit_buffer_impl<SIZE, 4>();
    case 5:
      return new compiled_time_bit_buffer_impl<SIZE, 5>();
    default:
      throw std::invalid_argument("Invalid bit size");
  }
}

// Constants
// --------------------------------------------------------------

constexpr size_t BUFFER_SIZE = 64 * 64 * 64;
constexpr size_t N_WRITE_OPS = 1024;
constexpr size_t N_READ_OPS = 1024;

// Generate random write operations
// --------------------------------------------------------------
std::vector<std::pair<size_t, uint8_t>>& generate_uniform_write_ops(
    size_t bit_size) {
  static bool initialized = false;
  static std::vector<std::pair<size_t, uint8_t>> write_ops(N_WRITE_OPS);

  if (initialized) {
    return write_ops;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> tdist(0, BUFFER_SIZE - 1);
  std::uniform_int_distribution<uint32_t> vdist(
      0, (1 << (uint64_t(1) << bit_size)) - 1);

  for (size_t i = 0; i < N_WRITE_OPS; ++i) {
    size_t index = tdist(gen);
    uint32_t value = vdist(gen);
    write_ops[i] = {index, value};
  }

  initialized = true;

  return write_ops;
}

// std::vector<std::pair<size_t, uint8_t>>& generate_local_write_ops(
//     size_t bit_size) {
//   static bool initialized = false;
//   static std::vector<std::pair<size_t, uint8_t>> write_ops(N_WRITE_OPS);
// }

// Benchmarking
// --------------------------------------------------------------
static void BM_RuntimeBitBufferUniformWrite(benchmark::State& state) {
  size_t bit_size = state.range(0);
  runtime_bit_buffer buffer(BUFFER_SIZE, bit_size);

  auto& write_ops = generate_uniform_write_ops(bit_size);

  for (auto _ : state) {
    for (const auto& op : write_ops) {
      buffer.set(op.first, op.second);
    }

    benchmark::DoNotOptimize(buffer);
  }
}

static void BM_CompileTimeBitBufferUniformWrite(benchmark::State& state) {
  size_t bit_size = state.range(0);
  auto buffer = create_compiled_time_bit_buffer<BUFFER_SIZE>(bit_size);

  auto& write_ops = generate_uniform_write_ops(bit_size);

  for (auto _ : state) {
    for (const auto& op : write_ops) {
      buffer->set(op.first, op.second);
    }

    benchmark::DoNotOptimize(buffer);
  }

  delete buffer;
}

BENCHMARK(BM_RuntimeBitBufferUniformWrite)
    ->DenseRange(0, 5, 5)
    ->Unit(benchmark::kMicrosecond);

// BENCHMARK(BM_CompileTimeBitBufferUniformWrite)
//     ->DenseRange(0, 5, 1)
//     ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();