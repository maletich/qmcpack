//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#include "catch.hpp"

#include <memory>
#include <vector>
#include <iostream>
#include "OMPTarget/OMPallocator.hpp"
#include "OMPTarget/ompBLAS.hpp"
#include <OhmmsPETE/OhmmsVector.h>
#include <OhmmsPETE/OhmmsMatrix.h>
#include <CPU/BLAS.hpp>

// Timer
#include <chrono>
#include <string>

namespace qmcplusplus
{

template<typename T>
void test_gemv(const int M_b, const int N_b, const char trans)
{
  const int M = trans == 'T' ? M_b : N_b;
  const int N = trans == 'T' ? N_b : M_b;

  using vec_t = Vector<T, OMPallocator<T>>;
  using mat_t = Matrix<T, OMPallocator<T>>;

  ompBLAS::ompBLAS_handle handle;

  vec_t A(N);        // Input vector
  mat_t B(M_b, N_b); // Input matrix
  vec_t C(M);        // Result vector ompBLAS
  vec_t D(M);        // Result vector BLAS

  // Fill data
  for (int i = 0; i < N; i++)
    A[i] = i;

  for (int j = 0; j < M_b; j++)
    for (int i = 0; i < N_b; i++)
      B[j][i] = i + j * 2;

  // Fill C and D with 0
  for (int i = 0; i < M; i++)
    C[i] = D[i] = T(0);

  A.updateTo();
  B.updateTo();

  T alpha(1);
  T beta(0);

  // in Fortran, B[M][N] is viewed as B^T
  // when trans == 'T', the actual calculation is B * A[N] = C[M]
  // when trans == 'N', the actual calculation is B^T * A[M] = C[N]
  ompBLAS::gemv(handle, trans, N_b, M_b, alpha, B.device_data(), N_b, A.device_data(), 1, beta, C.device_data(), 1);

  if (trans == 'T')
    BLAS::gemv_trans(M_b, N_b, B.data(), A.data(), D.data());
  else
    BLAS::gemv(M_b, N_b, B.data(), A.data(), D.data());

  C.updateFrom();

  bool are_same = true;
  int index     = 0;
  do
  {
    are_same = C[index] == D[index];
    CHECK(C[index] == D[index]);
    index++;
  } while (are_same == true && index < M);
}

template<typename T>
void test_gemv_batched(const int M_b, const int N_b, const char trans, const int batch_count)
{
  const int M = trans == 'T' ? M_b : N_b;
  const int N = trans == 'T' ? N_b : M_b;

  using vec_t = Vector<T, OMPallocator<T>>;
  using mat_t = Matrix<T, OMPallocator<T>>;

  ompBLAS::ompBLAS_handle handle;

  // Create input vector
  std::vector<vec_t> As;
  Vector<const T*, OMPallocator<const T*>> Aptrs;

  // Create input matrix
  std::vector<mat_t> Bs;
  Vector<const T*, OMPallocator<const T*>> Bptrs;

  // Create output vector (ompBLAS)
  std::vector<vec_t> Cs;
  Vector<T*, OMPallocator<T*>> Cptrs;

  // Create output vector (BLAS)
  std::vector<vec_t> Ds;
  Vector<T*, OMPallocator<T*>> Dptrs;

  // Resize pointer vectors
  Aptrs.resize(batch_count);
  Bptrs.resize(batch_count);
  Cptrs.resize(batch_count);
  Dptrs.resize(batch_count);

  // Resize data vectors
  As.resize(batch_count);
  Bs.resize(batch_count);
  Cs.resize(batch_count);
  Ds.resize(batch_count);

  // Fill data
  for (int batch = 0; batch < batch_count; batch++)
  {
    handle = batch;

    As[batch].resize(N);
    Aptrs[batch] = As[batch].device_data();

    Bs[batch].resize(M_b, N_b);
    Bptrs[batch] = Bs[batch].device_data();

    Cs[batch].resize(M);
    Cptrs[batch] = Cs[batch].device_data();

    Ds[batch].resize(M);
    Dptrs[batch] = Ds[batch].data();

    for (int i = 0; i < N; i++)
      As[batch][i] = i;

    for (int j = 0; j < M_b; j++)
      for (int i = 0; i < N_b; i++)
        Bs[batch][j][i] = i + j * 2;

    for (int i = 0; i < M; i++)
      Cs[batch][i] = Ds[batch][i] = T(0);

    As[batch].updateTo();
    Bs[batch].updateTo();
  }

  Aptrs.updateTo();
  Bptrs.updateTo();
  Cptrs.updateTo();

  // Run tests
  Vector<T, OMPallocator<T>> alpha;
  alpha.resize(batch_count);
  Vector<T, OMPallocator<T>> beta;
  beta.resize(batch_count);

  for (int batch = 0; batch < batch_count; batch++)
  {
    alpha[batch] = T(1);
    beta[batch]  = T(0);
  }

  alpha.updateTo();
  beta.updateTo();

  ompBLAS::gemv_batched(handle, trans, N_b, M_b, alpha.device_data(), Bptrs.device_data(), N_b, Aptrs.device_data(), 1,
                        beta.device_data(), Cptrs.device_data(), 1, batch_count);


  for (int batch = 0; batch < batch_count; batch++)
  {
    if (trans == 'T')
      BLAS::gemv_trans(M_b, N_b, Bs[batch].data(), As[batch].data(), Ds[batch].data());
    else
      BLAS::gemv(M_b, N_b, Bs[batch].data(), As[batch].data(), Ds[batch].data());
  }

  for (int batch = 0; batch < batch_count; batch++)
    Cs[batch].updateFrom();


  // Check results
  for (int batch = 0; batch < batch_count; batch++)
  {
    bool are_same = true;
    int index     = 0;
    do
    {
      are_same = Cs[batch][index] == Ds[batch][index];
      CHECK(Cs[batch][index] == Ds[batch][index]);
      index++;
    } while (are_same == true && index < M);
  }
}

template<typename T>
void test_gemm(const int N, const int M)
{}

template<typename T>
void timer(const int M_b, const int N_b, const char trans, const int batch_count, const char type)
{
  // Set handle
  ompBLAS::ompBLAS_handle handle = 1;

  // Adjust M and N
  const int M = trans == 'T' ? M_b : N_b;
  const int N = trans == 'T' ? N_b : M_b;

  if (type == 'n')
  {
    using vec_t = Vector<T, OMPallocator<T>>;
    using mat_t = Matrix<T, OMPallocator<T>>;

    vec_t A(N);        // Input vector
    mat_t B(M_b, N_b); // Input matrix
    vec_t C(M);        // Result vector ompBLAS
    vec_t D(M);        // Result vector BLAS

    // Fill data
    for (int i = 0; i < N; i++)
      A[i] = i;

    for (int j = 0; j < M_b; j++)
      for (int i = 0; i < N_b; i++)
        B[j][i] = i + j * 2;

    // Fill C and D with 0
    for (int i = 0; i < M; i++)
      C[i] = T(0);

    A.updateTo();
    B.updateTo();

    T alpha(1);
    T beta(0);

    auto start = std::chrono::system_clock::now();
    // Run tests in here
    for (int i = 0; i < batch_count; i++)
    {
      ompBLAS::gemv(handle, trans, N_b, M_b, alpha, B.device_data(), N_b, A.device_data(), 1, beta, C.device_data(), 1);
    }
    // End of test
    auto end = std::chrono::system_clock::now();

    typedef std::chrono::duration<float, std::milli> duration;
    duration elapsed_seconds = end - start;

    std::cout << "Non-batched elapsed time: " << elapsed_seconds.count() << "ms" << std::endl;
  }
  else if (type == 'b')
  {
    using vec_t = Vector<T, OMPallocator<T>>;
    using mat_t = Matrix<T, OMPallocator<T>>;

    ompBLAS::ompBLAS_handle handle;

    // Create input vector
    std::vector<vec_t> As;
    Vector<const T*, OMPallocator<const T*>> Aptrs;

    // Create input matrix
    std::vector<mat_t> Bs;
    Vector<const T*, OMPallocator<const T*>> Bptrs;

    // Create output vector (ompBLAS)
    std::vector<vec_t> Cs;
    Vector<T*, OMPallocator<T*>> Cptrs;

    // Resize pointer vectors
    Aptrs.resize(batch_count);
    Bptrs.resize(batch_count);
    Cptrs.resize(batch_count);

    // Resize data vectors
    As.resize(batch_count);
    Bs.resize(batch_count);
    Cs.resize(batch_count);

    // Fill data
    for (int batch = 0; batch < batch_count; batch++)
    {
      handle = batch;

      As[batch].resize(N);
      Aptrs[batch] = As[batch].device_data();

      Bs[batch].resize(M_b, N_b);
      Bptrs[batch] = Bs[batch].device_data();

      Cs[batch].resize(M);
      Cptrs[batch] = Cs[batch].device_data();

      for (int i = 0; i < N; i++)
        As[batch][i] = i;

      for (int j = 0; j < M_b; j++)
        for (int i = 0; i < N_b; i++)
          Bs[batch][j][i] = i + j * 2;

      for (int i = 0; i < M; i++)
        Cs[batch][i] = T(0);

      As[batch].updateTo();
      Bs[batch].updateTo();
    }

    Aptrs.updateTo();
    Bptrs.updateTo();
    Cptrs.updateTo();

    Vector<T, OMPallocator<T>> alpha;
    alpha.resize(batch_count);
    Vector<T, OMPallocator<T>> beta;
    beta.resize(batch_count);

    for (int batch = 0; batch < batch_count; batch++)
    {
      alpha[batch] = T(1);
      beta[batch]  = T(0);
    }

    alpha.updateTo();
    beta.updateTo();

    auto start = std::chrono::system_clock::now();
    // Start test
    ompBLAS::gemv_batched(handle, trans, N_b, M_b, alpha.device_data(), Bptrs.device_data(), N_b, Aptrs.device_data(),
                          1, beta.device_data(), Cptrs.device_data(), 1, batch_count);
    // End of test
    auto end = std::chrono::system_clock::now();

    typedef std::chrono::duration<float, std::milli> duration;
    duration elapsed_seconds = end - start;

    std::cout << "Batched elapsed time: " << elapsed_seconds.count() << "ms" << std::endl;
  }
}

TEST_CASE("OmpBLAS gemv", "[OMP]")
{
  const int N           = 100;
  const int M           = 200;
  const int batch_count = 14;

  // NOTRNS NOT IMPL
  /*
      std::cout << "Testing NOTRANS gemv" << std::endl;
      test_gemv<float>(N, 'N');
      test_gemv<double>(N, 'N');
      #if defined(QMC_COMPLEX)
      test_gemv<std::complex<float>>(N, 'N');
      test_gemv<std::complex<double>>(N, 'N');
      #endif
    */

  const int max = 1000;
  // Timer test
  for (int i = 0; i <= max; i += 100)
  {
    std::cout << "Batch count: " << i << std::endl;
    timer<double>(M, N, 'T', i, 'n');
    timer<double>(M, N, 'T', i, 'b');
  }
  // Non-batched test
  std::cout << "Testing TRANS gemv" << std::endl;
  test_gemv<float>(M, N, 'T');
  test_gemv<double>(M, N, 'T');
#if defined(QMC_COMPLEX)
  test_gemv<std::complex<float>>(N, M, 'T');
  test_gemv<std::complex<double>>(N, M, 'T');
#endif
  // Batched Test
  std::cout << "Testing TRANS gemv_batched" << std::endl;
  test_gemv_batched<float>(M, N, 'T', batch_count);
  test_gemv_batched<double>(M, N, 'T', batch_count);
#if defined(QMC_COMPLEX)
  test_gemv<std::complex<float>>(M, N, 'T', batch_count);
  test_gemv<std::complex<double>>(M, N, 'T', batch_count);
#endif
}
} // namespace qmcplusplus
