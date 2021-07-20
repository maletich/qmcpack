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

#define N 100

#include "catch.hpp"

#include <memory>
#include <vector>
#include <iostream>
#include "OMPTarget/OMPallocator.hpp"
#include "OhmmsPETE/OhmmsVector.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "Numerics/OhmmsBlas.h"

/*
ompBLAS_status gemv(ompBLAS_handle&    handle, // dummy
                    const char         trans, // "T" or "t"
                    const int          m, // number of rows in matrix A
                    const int          n, // number of cols in matrix A
                    const float        alpha, // scalar alpha (1.0)
                    const float* const A, // Matrix pointer
                    const int          lda, // First dimension of A
                    const float* const x, // Vector pointer
                    const int          incx, // increment for vector's elements
                    const float        beta, // scalar beta (1.0)
                    float* const       y, // Result vector
                    const int          incy); // increment for Y
*/

namespace qmcplusplus
{
TEST_CASE("OMPmath", "[OMP]")
{
 using vec_t = Vector<double, OMPallocator<double>>;
 using mat_t = Matrix<double, OMPallocator<double>>;

 PRAGMA_OFFLOAD("omp target teams distribute map(to : B[:N * N], A[:N]) map(from : C[:N], D[:N])")
  for(int i = 0; i < N; i++)
    {
      vec_t A(i); // Input vector
      mat_t B(i); // Input matrix
      vec_t C(i); // Result vector (test)
      vec_t D(i) // Result vector BLAS
      
      gemv(dummy, "T", i, i, 1.0, B, 1, A, 1, 1.0, C, 1); // tests omp gemv
      BLAS.gemv(i, i, B, A, D); // BLAS gemv
      REQUIRE(std::equal(C, D) == true);  // Compares results and determines if each loop passes
    }
}

} // namespace qmcplusplus
