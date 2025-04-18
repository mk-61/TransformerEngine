#include <gtest/gtest.h>
#include <mpi.h>
#include <transformer_engine/comm_gemm.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../test_common.h"
#include "common.h"

#define CHECK_MPI(expr)                                              \
  do {                                                               \
    int err = (expr);                                                \
    if (err != MPI_SUCCESS) {                                        \
      char err_str[MPI_MAX_ERROR_STRING + 1]{};                      \
      int _len{};                                                    \
      MPI_Error_string(err, err_str, &_len);                         \
      EXPECT_TRUE(false) << "MPI error: " << err << ": " << err_str; \
    }                                                                \
  } while (false)

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  CHECK_MPI(MPI_Init(&argc, &argv));
  auto ret = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return ret;
}

struct Params {
  transformer_engine::DType dtype;
  bool transa;
  bool transb;
  size_t m;
  size_t n;
  size_t k;
};

template <typename T>
std::vector<T> CopyLocalMatrix(const std::vector<T>& data, size_t mstart, size_t nstart,
                               size_t msize, size_t nsize, size_t ld) {
  std::vector<T> ret(msize * nsize);
  size_t dst = 0;
  for (size_t j = nstart; j < nstart + nsize; ++j) {
    for (size_t i = mstart; i < mstart + msize; ++i) {
      ret[dst++] = data[j * ld + i];
    }
  }
  return ret;
}

template <typename T>
test::Tensor MakeTensorFromData(const std::vector<T>& data, size_t mstart, size_t nstart,
                                size_t msize, size_t nsize, size_t ld,
                                transformer_engine::DType dtype) {
  auto values = CopyLocalMatrix(data, mstart, nstart, msize, nsize, ld);
  test::Tensor ret("", {nsize, msize}, dtype, true, true);
  NVTE_CHECK_CUDA(cudaMemcpy(ret.rowwise_dptr(), values.data(), values.size() * sizeof values[0],
                             cudaMemcpyDefault));
  return ret;
}

test::Tensor MakeTensor(size_t msize, size_t nsize, transformer_engine::DType dtype) {
  return test::Tensor("", {nsize, msize}, dtype, true, true);
}

struct PatternDims {
  size_t a_rows_start;
  size_t a_rows_num;
  size_t a_cols_start;
  size_t a_cols_num;
  size_t b_rows_start;
  size_t b_rows_num;
  size_t b_cols_start;
  size_t b_cols_num;
  size_t d_rows_start;
  size_t d_rows_num;
  size_t d_cols_start;
  size_t d_cols_num;
};

class CommGemmTest : public ::testing::TestWithParam<Params> {
 protected:
  CommGemmTest() {
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks_));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    int local_device = rank_;
    NVTE_CHECK_CUDA(cudaSetDevice(rank_));
    ctx_ = nvte_comm_gemm_ctx_create(nranks_, rank_, local_device);
  }
  ~CommGemmTest() { nvte_comm_gemm_ctx_destroy(ctx_); }

  virtual PatternDims DistributeTensors(size_t m, size_t n, size_t k) = 0;

  virtual void CommGemm(size_t m, size_t n, size_t k, const NVTETensor a, const NVTETensor b,
                        const NVTETensor d, const NVTETensor bias, const NVTETensor pre_act_out,
                        bool transa, bool transb, bool grad, bool accumulate, int comm_sm_count,
                        cudaStream_t stream) = 0;

  template <typename T>
  void Run(bool transa, bool transb, size_t m, size_t n, size_t k,
           transformer_engine::DType dtype) {
    cudaStream_t stream{};
    NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

    std::mt19937 rng(12);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<T> adata(m * k);
    std::generate(adata.begin(), adata.end(), [&rng, &dist] { return static_cast<T>(dist(rng)); });
    std::vector<T> bdata(k * n);
    std::generate(bdata.begin(), bdata.end(), [&rng, &dist] { return static_cast<T>(dist(rng)); });

    auto ga = transa ? MakeTensorFromData<T>(adata, 0, 0, k, m, k, dtype)
                     : MakeTensorFromData<T>(adata, 0, 0, m, k, m, dtype);
    auto gb = transb ? MakeTensorFromData<T>(bdata, 0, 0, n, k, n, dtype)
                     : MakeTensorFromData<T>(bdata, 0, 0, k, n, k, dtype);
    auto gd = MakeTensor(m, n, dtype);

    auto dims = DistributeTensors(m, n, k);
    auto a = transa ? MakeTensorFromData<T>(adata, dims.a_rows_start, dims.a_cols_start,
                                            dims.a_rows_num, dims.a_cols_num, k, dtype)
                    : MakeTensorFromData<T>(adata, dims.a_cols_start, dims.a_rows_start,
                                            dims.a_cols_num, dims.a_rows_num, m, dtype);
    auto b = transb ? MakeTensorFromData<T>(bdata, dims.b_cols_start, dims.b_rows_start,
                                            dims.b_cols_num, dims.b_rows_num, n, dtype)
                    : MakeTensorFromData<T>(bdata, dims.b_rows_start, dims.b_cols_start,
                                            dims.b_rows_num, dims.b_cols_num, k, dtype);
    auto d = MakeTensor(dims.d_rows_num, dims.d_cols_num, dtype);

    transformer_engine::Tensor bias;
    transformer_engine::Tensor pre_act_out;
    bool grad = false;
    bool accumulate = false;
    CommGemm(m, n, k, a.data(), b.data(), d.data(), &bias, &pre_act_out, transa, transb, grad,
             accumulate, 0 /*comm_sm_count*/, stream);
    auto workspace = MakeTensor(1, 32 << 20, transformer_engine::DType::kByte);
    nvte_cublas_gemm(ga.data(), gb.data(), gd.data(), &bias, &pre_act_out, transa, transb, grad,
                     workspace.data(), accumulate, false /* use_split_accumulator */,
                     0 /* math_sm_count */, stream);
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
    std::vector<T> out(dims.d_rows_num * dims.d_cols_num);
    NVTE_CHECK_CUDA(
        cudaMemcpy(out.data(), d.rowwise_dptr(), out.size() * sizeof out[0], cudaMemcpyDefault));
    std::vector<T> out_golden_global(m * n);
    NVTE_CHECK_CUDA(cudaMemcpy(out_golden_global.data(), gd.rowwise_dptr(),
                               out_golden_global.size() * sizeof out_golden_global[0],
                               cudaMemcpyDefault));

    auto out_golden = CopyLocalMatrix(out_golden_global, dims.d_rows_start, dims.d_cols_start,
                                      dims.d_rows_num, dims.d_cols_num, m);
    NVTE_CHECK(out.size() == out_golden.size());
    // TODO: use cuda::std::numeric_limits<T>::epsilon(), when it's not broken.
    const double dtype_epsilon = 1e-3;
    const auto atol = dtype_epsilon * k;
    for (size_t i = 0; i < out.size(); ++i) {
      if (rand() % 100 < 3) {
        std::cerr << "== AT " << rank_ << ": " << i << ": " << static_cast<double>(out[i]) << " "
                  << static_cast<double>(out_golden[i]) << std::endl;
      }
      EXPECT_NEAR(static_cast<double>(out[i]), static_cast<double>(out_golden[i]), atol);
    }
  }

  CommGemmCtx* ctx_{};
  int nranks_{};
  int rank_{};
};

struct AgGemmTest : public CommGemmTest {
  PatternDims DistributeTensors(size_t m, size_t n, size_t k) override {
    auto a_cols_num = static_cast<size_t>(nvte_comm_gemm_numroc(ctx_, m));
    auto b_cols_num = static_cast<size_t>(nvte_comm_gemm_numroc(ctx_, n));

    size_t a_cols_start{};
    size_t b_cols_start{};
    MPI_Exscan(&a_cols_num, &a_cols_start, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(&b_cols_num, &b_cols_start, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    return PatternDims{
        .a_rows_start = 0,
        .a_rows_num = k,
        .a_cols_start = a_cols_start,
        .a_cols_num = a_cols_num,
        .b_rows_start = 0,
        .b_rows_num = k,
        .b_cols_start = b_cols_start,
        .b_cols_num = b_cols_num,
        .d_rows_start = a_cols_start,
        .d_rows_num = a_cols_num,
        .d_cols_start = 0,
        .d_cols_num = n,
    };
  }

  void CommGemm(size_t m, size_t n, size_t k, const NVTETensor a, const NVTETensor b,
                const NVTETensor d, const NVTETensor bias, const NVTETensor pre_act_out,
                bool transa, bool transb, bool grad, bool accumulate, int comm_sm_count,
                cudaStream_t stream) override {
    nvte_all_gather_gemm(ctx_, m, n, k, a, b, d, bias, pre_act_out, transa, transb, grad,
                         accumulate, comm_sm_count, stream);
  }
};

struct GemmRsTest : public CommGemmTest {
  PatternDims DistributeTensors(size_t m, size_t n, size_t k) override {
    auto rows_num = static_cast<size_t>(nvte_comm_gemm_numroc(ctx_, k));
    auto d_cols_num = static_cast<size_t>(nvte_comm_gemm_numroc(ctx_, n));

    size_t rows_start{};
    size_t d_cols_start{};
    MPI_Exscan(&rows_num, &rows_start, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(&d_cols_num, &d_cols_start, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    return PatternDims{
        .a_rows_start = rows_start,
        .a_rows_num = rows_num,
        .a_cols_start = 0,
        .a_cols_num = m,
        .b_rows_start = rows_start,
        .b_rows_num = rows_num,
        .b_cols_start = 0,
        .b_cols_num = n,
        .d_rows_start = 0,
        .d_rows_num = m,
        .d_cols_start = d_cols_start,
        .d_cols_num = d_cols_num,
    };
  }

  void CommGemm(size_t m, size_t n, size_t k, const NVTETensor a, const NVTETensor b,
                const NVTETensor d, const NVTETensor bias, const NVTETensor pre_act_out,
                bool transa, bool transb, bool grad, bool accumulate, int comm_sm_count,
                cudaStream_t stream) override {
    nvte_gemm_reduce_scatter(ctx_, m, n, k, a, b, d, bias, pre_act_out, transa, transb, grad,
                             accumulate, comm_sm_count, stream);
  }
};

struct GemmArTest : public CommGemmTest {
  PatternDims DistributeTensors(size_t m, size_t n, size_t k) override {
    auto rows_num = static_cast<size_t>(nvte_comm_gemm_numroc(ctx_, k));

    size_t rows_start{};
    MPI_Exscan(&rows_num, &rows_start, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    return PatternDims{
        .a_rows_start = rows_start,
        .a_rows_num = rows_num,
        .a_cols_start = 0,
        .a_cols_num = m,
        .b_rows_start = rows_start,
        .b_rows_num = rows_num,
        .b_cols_start = 0,
        .b_cols_num = n,
        .d_rows_start = 0,
        .d_rows_num = m,
        .d_cols_start = 0,
        .d_cols_num = n,
    };
  }

  void CommGemm(size_t m, size_t n, size_t k, const NVTETensor a, const NVTETensor b,
                const NVTETensor d, const NVTETensor bias, const NVTETensor pre_act_out,
                bool transa, bool transb, bool grad, bool accumulate, int comm_sm_count,
                cudaStream_t stream) override {
    nvte_gemm_all_reduce(ctx_, m, n, k, a, b, d, bias, pre_act_out, transa, transb, grad,
                         accumulate, comm_sm_count, stream);
  }
};

TEST_P(AgGemmTest, AgGemm) {
  auto [dtype, transa, transb, m, n, k] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(dtype, DType,
                                        { Run<DType>(transa, transb, m, n, k, dtype); });
}

TEST_P(GemmRsTest, GemmRs) {
  auto [dtype, transa, transb, m, n, k] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(dtype, DType,
                                        { Run<DType>(transa, transb, m, n, k, dtype); });
}

TEST_P(GemmArTest, GemmAr) {
  auto [dtype, transa, transb, m, n, k] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(dtype, DType,
                                        { Run<DType>(transa, transb, m, n, k, dtype); });
}

using transformer_engine::DType;

std::string ParamSuffix(const testing::TestParamInfo<Params>& info) {
  const auto [dtype, transa, transb, m, n, k] = info.param;
  std::ostringstream ss;
  ss << to_string(dtype) << "_" << (transa ? "T" : "N") << (transb ? "T" : "N") << "_" << m << "x"
     << n << "x" << k;
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(AgGemm, AgGemmTest,
                         testing::Values(Params{DType::kFloat16, false, false, 256, 128, 64},
                                         Params{DType::kFloat16, false, true, 256, 128, 64},
                                         Params{DType::kFloat16, true, false, 256, 128, 64},
                                         Params{DType::kBFloat16, false, false, 256, 128, 64},
                                         Params{DType::kBFloat16, false, true, 256, 128, 64},
                                         Params{DType::kBFloat16, true, false, 256, 128, 64},
                                         Params{DType::kFloat8E4M3, false, false, 256, 128, 64},
                                         Params{DType::kFloat8E4M3, false, true, 256, 128, 64},
                                         Params{DType::kFloat8E4M3, true, false, 256, 128, 64}),
                         &ParamSuffix);

INSTANTIATE_TEST_SUITE_P(
    GemmRs, GemmRsTest,
    testing::Values(Params{transformer_engine::DType::kFloat16, true, false, 64, 128, 256},
                    Params{transformer_engine::DType::kFloat8E4M3, true, false, 64, 128, 256}),
    &ParamSuffix);

INSTANTIATE_TEST_SUITE_P(GemmAr, GemmArTest,
                         testing::Values(Params{transformer_engine::DType::kFloat16, true, false,
                                                64, 64 * 4, 64 * 4}),
                         &ParamSuffix);
