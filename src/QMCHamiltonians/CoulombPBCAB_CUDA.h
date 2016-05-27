//////////////////////////////////////////////////////////////////
// (c) Copyright 2010-  by Ken Esler and Jeongnim Kim
//////////////////////////////////////////////////////////////////
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: esler@uiuc.edu
//
// Supported by
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_COULOMBPBCAB_CUDA_H
#define QMCPLUSPLUS_COULOMBPBCAB_CUDA_H
#include "QMCHamiltonians/CoulombPBCAB.h"
#include "QMCHamiltonians/CudaCoulomb.h"


namespace qmcplusplus
{
struct CoulombPBCAB_CUDA : public CoulombPBCAB
{
  //////////////////////////////////
  // Vectorized evaluation on GPU //
  //////////////////////////////////
  //// Short-range part
  int NumIons, NumElecs, NumElecGroups, NumIonSpecies;
  ParticleSet &ElecRef, &IonRef;
  std::vector<int> IonFirst, IonLast;
  // This is indexed by the ion species
  std::vector<TextureSpline*> SRSplines;
  TextureSpline *V0Spline;
  gpu::device_vector<CUDA_COULOMB_PRECISION>  SumGPU;
  gpu::host_vector<CUDA_COULOMB_PRECISION>  SumHost;
  gpu::device_vector<CUDA_PRECISION>  IGPU;
  gpu::device_vector<CUDA_COULOMB_PRECISION>  L, Linv;
  //// Long-range part
  int Numk;
  gpu::device_vector<CUDA_COULOMB_PRECISION> kpointsGPU;
  gpu::device_vector<int>            kshellGPU;
  // This has the same lengths as KshellGPU
  gpu::device_vector<CUDA_COULOMB_PRECISION> FkGPU;
  // The first vector index is the species number
  // Complex, stored as float2
  // This is for the electrons -- one per walker
  gpu::device_vector<CUDA_COULOMB_PRECISION*>  RhoklistGPU;
  gpu::host_vector<CUDA_COULOMB_PRECISION*>  RhoklistHost;
  // This stores rho_k for the electrons in one big array
  gpu::device_vector<CUDA_COULOMB_PRECISION> RhokElecGPU;

  std::vector<PosType> SortedIons;
  // This stores rho_k for the ions.  Index is species number
  std::vector<gpu::device_vector<CUDA_COULOMB_PRECISION> > RhokIonsGPU;
  void setupLongRangeGPU();

  void add(int groupID, RadFunctorType* ppot);

  void initBreakup(ParticleSet& P);

  void addEnergy(MCWalkerConfiguration &W,
                 std::vector<RealType> &LocalEnergy);

  QMCHamiltonianBase* makeClone(ParticleSet& qp, TrialWaveFunction& psi);

  CoulombPBCAB_CUDA(ParticleSet& ions, ParticleSet& elns,
                    bool cloning=false);

};
}

#endif
