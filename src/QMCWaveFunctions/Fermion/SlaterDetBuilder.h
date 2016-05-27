//////////////////////////////////////////////////////////////////
// (c) Copyright 2003  by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: jnkim@ncsa.uiuc.edu
//   Tel:    217-244-6319 (NCSA) 217-333-3324 (MCC)
//
// Supported by
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_LCORBITALSETBUILDER_H
#define QMCPLUSPLUS_LCORBITALSETBUILDER_H

#include <vector>
#include "QMCWaveFunctions/OrbitalBuilderBase.h"
#include "QMCWaveFunctions/BasisSetFactory.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDeterminant.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDeterminantFast.h"
#include "QMCWaveFunctions/Fermion/ci_configuration.h"
#include "QMCWaveFunctions/Fermion/ci_configuration2.h"
#include "QMCWaveFunctions/Fermion/BackflowTransformation.h"
#include "QMCWaveFunctions/Fermion/BackflowBuilder.h"
namespace qmcplusplus
{

/** derived class from OrbitalBuilderBase
 *
 * Builder SlaterDeterminant with LCOrbitalSet
 */
class SlaterDetBuilder: public OrbitalBuilderBase
{

public:

  typedef SlaterDet SlaterDeterminant_t;
  typedef MultiSlaterDeterminant MultiSlaterDeterminant_t;
  typedef DiracDeterminantBase Det_t;
  /** constructor
   * \param els reference to the electrons
   * \param psi reference to the wavefunction
   * \param ions reference to the ions
   */
  SlaterDetBuilder(ParticleSet& els, TrialWaveFunction& psi, PtclPoolType& psets);

  ~SlaterDetBuilder();

  /** initialize the Antisymmetric wave function for electrons
   *@param cur the current xml node
   *
   */
  bool put(xmlNodePtr cur);

private:

  ///reference to a PtclPoolType
  PtclPoolType& ptclPool;
  BasisSetFactory* myBasisSetFactory;
  SlaterDeterminant_t* slaterdet_0;
  MultiSlaterDeterminant_t* multislaterdet_0;
  MultiSlaterDeterminantFast* multislaterdetfast_0;

  bool UseBackflow;
  BackflowTransformation *BFTrans;

  /** process a determinant element
   * @param cur xml node
   * @param firstIndex index of the determinant
   * @return firstIndex+number of orbitals
   */
  bool putDeterminant(xmlNodePtr cur, int firstIndex);

  bool createMSD(MultiSlaterDeterminant* multiSD, xmlNodePtr cur);

  bool createMSDFast(MultiSlaterDeterminantFast* multiSD, xmlNodePtr cur);

  bool readDetList(xmlNodePtr cur, std::vector<ci_configuration>& uniqueConfg_up, std::vector<ci_configuration>& uniqueConfg_dn, std::vector<int>& C2node_up, std::vector<int>& C2node_dn, std::vector<std::string>& CItags, std::vector<RealType>& coeff, bool& optimizeCI, int nels_up, int nels_dn, std::vector<RealType>& CSFcoeff, std::vector<int>& DetsPerCSF, std::vector<RealType>& CSFexpansion, bool& usingCSF);

};
}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$
 ***************************************************************************/
