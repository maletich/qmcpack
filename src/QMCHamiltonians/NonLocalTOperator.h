//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file NonLocalTOperator.h
 * @brief Declaration of NonLocalTOperator
 *
 * NonLocalTOperator has the off-diagonal transition probability matrix.
 */
#ifndef QMCPLUSPLUS_NONLOCALTRANSITIONOPERATOR_H
#define QMCPLUSPLUS_NONLOCALTRANSITIONOPERATOR_H

#include "Configuration.h"
#include <RandomGenerator.h>

namespace qmcplusplus
{
class ParticleSet;
class TrialWaveFunction;
class NonLocalECPotential;

struct NonLocalData : public QMCTraits
{
  IndexType PID;
  RealType Weight;
  PosType Delta;
  inline NonLocalData() : PID(-1), Weight(1.0) {}
  inline NonLocalData(IndexType id, RealType w, const PosType& d) : PID(id), Weight(w), Delta(d) {}
};

struct NonLocalTOperator
{
  typedef NonLocalData::RealType RealType;
  typedef NonLocalData::PosType PosType;

  /// Tmove options
  enum class Scheme
  {
    OFF = 0, // no Tmove
    V0,      // M. Casula, PRB 74, 161102(R) (2006)
    V1,      // version 1, M. Casula et al., JCP 132, 154113 (2010)
    V3,      // an approximation to version 1 but much faster.
  };

  std::vector<NonLocalData> Txy;
  std::vector<std::vector<NonLocalData>> Txy_by_elec;

  NonLocalTOperator(size_t N);

  inline int size() const { return Txy.size(); }

  /** replacement for put because wouldn't it be cool to know what the classes configuration actually
   *  is.
   */
  void thingsThatShouldBeInMyConstructor(const std::string& non_local_move_option,
                                        const double tau,
                                        const double alpha,
                                        const double gamma);
  /** initialize the parameters */
  void put(xmlNodePtr cur);

  /** reset Txy for a new set of non-local moves
   *
   * Txy[0] is always 1 corresponding to the diagonal(no) move
   */
  void reset();

  /** select the move for a given probability
   * @param prob value [0,1)
   * @param txy a given Txy collection
   * @return pointer to NonLocalData
   */
  const NonLocalData* selectMove(RealType prob, std::vector<NonLocalData>& txy) const;

  /** select the move for a given probability using internal Txy
   * @param prob value [0,1)
   * @return pointer to NonLocalData
   */
  inline const NonLocalData* selectMove(RealType prob) { return selectMove(prob, Txy); }

  /** select the move for a given probability using internal Txy_by_elec
   * @param prob value [0,1)
   * @param iel reference electron
   * @return pointer to NonLocalData
   */
  inline const NonLocalData* selectMove(RealType prob, int iel) { return selectMove(prob, Txy_by_elec[iel]); }

  /** sort all the Txy elements by electron */
  void group_by_elec();

  Scheme getScheme() const { return scheme_; }

  /** make non local moves with particle-by-particle moves
   * @param P particle set
   * @return the number of accepted moves
   */
  static int makeNonLocalMoves(ParticleSet& P, TrialWaveFunction& Psi, NonLocalECPotential& nlpp, RandomGenerator_t& myRNG);

private:
  /// tmove selection
  Scheme scheme_;
  /// number of electrons
  const size_t Nelec;

  // parameters
  RealType Tau;
  RealType Alpha;
  RealType Gamma;
  RealType plusFactor;
  RealType minusFactor;
};

} // namespace qmcplusplus
#endif
