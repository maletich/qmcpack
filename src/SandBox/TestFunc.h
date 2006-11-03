#include "OhmmsPETE/TinyVector.h"
#include <cmath>
namespace qmcplusplus {

struct TestFunc {

  double k0,k1,k2;
  double d2factor;

  TestFunc(int nk0=1, int nk1=1, int nk2=1) {
    const double twopi = 8.0*std::atan(1.0);
    k0=twopi*static_cast<double>(nk0);
    k1=twopi*static_cast<double>(nk1);
    k2=twopi*static_cast<double>(nk2);
    d2factor = -(k0*k0+k1*k1+k2*k2);
  }

  //inline double operator()(const TinyVector<double,3>& pos) {
  //  return sin(k0*pos[0])*sin(k1*pos[1])*sin(k2*pos[2]);
  //}
  //inline double operator()(double x, double y, double z) {
  //  return sin(k0*x)*sin(k1*y)*sin(k2*z);
  //}
  //
  inline double f(const TinyVector<double,3>& pos) {
    return std::sin(k0*pos[0])*std::sin(k1*pos[1])*std::sin(k2*pos[2]);
  }

  inline double f(double x, double y, double z) {
    return std::sin(k0*x)*std::sin(k1*y)*std::sin(k2*z);
  }

  inline TinyVector<double,3> df(const TinyVector<double,3>& pos) {
    return TinyVector<double,3>(k0*std::cos(k0*pos[0])*std::sin(k1*pos[1])*std::sin(k2*pos[2]),
        k1*std::sin(k0*pos[0])*std::cos(k1*pos[1])*std::sin(k2*pos[2]),
        k2*std::sin(k0*pos[0])*std::sin(k1*pos[1])*std::cos(k2*pos[2]));
  }

  inline double d2f(const TinyVector<double,3>& pos) {
    return d2factor*f(pos);
  }

  inline double d2f(double x, double y, double z) {
    return d2factor*f(x,y,z);
  }

};

struct ComboFunc {

  std::vector<double> C;
  std::vector<TestFunc*> F;

  ComboFunc() {}
  ~ComboFunc() 
  {
    for(int i=0; i<F.size(); i++) delete F[i];
  }

  void push_back(double c, TestFunc* fn) { 
    C.push_back(c); 
    F.push_back(fn);
  }

  inline double f(double x, double y, double z) {
    double res=0;
    for(int i=0; i<C.size(); i++) res += C[i]*F[i]->f(x,y,z);
    return res;
  }

  inline TinyVector<double,3> df(const TinyVector<double,3>& pos) {
    TinyVector<double,3> res;
    for(int i=0; i<C.size(); i++) res += C[i]*F[i]->df(pos);
    return res;
  }

  inline double d2f(double x, double y, double z) {
    double res=0;
    for(int i=0; i<C.size(); i++) res += C[i]*F[i]->d2f(x,y,z);
    return res;
  }

  inline double f(const TinyVector<double,3>& pos) {
    return f(pos[0],pos[1],pos[2]);
  }

  inline double d2f(const TinyVector<double,3>& pos) {
    return d2f(pos[0],pos[1],pos[2]);
  }

};
}

