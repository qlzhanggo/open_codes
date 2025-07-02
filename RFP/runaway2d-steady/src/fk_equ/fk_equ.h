/**
 *  @file fk_equ.h
 *
 *  Created on: Nov 19, 2012
 *    Author: zehuag
 */

#ifndef FK_EQU_HPP_
#define FK_EQU_HPP_

#include <math.h>
#include <vector>
#include <memory>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>

#include "userdata.h"
#include "../mesh.h"
#include "diff_type.h"

#define square1(x) (x*x)
#define cubic(x) (x*x*x)
#define quad(x) (x*x*x*x)

class Simulation;

/**
 * fk_equ describes the fokker-plank equation
 */

class fk_equ
{
  friend class Simulation;

 protected:
  mesh      *my_mesh;
  PetscReal   Z;   // Charge of ion species
  PetscReal   rho, ctr, E0, sr; // rho/a, c\tau/R_0, E/E_c, \tau/\tau_s
  PetscReal   asp, q;  // aspect ratio, safety factor

  PetscReal ne, vte, Dpp;
  PetscReal   ra;
  PetscReal beta;  // one over base Coulomb logrithm
  PetscReal imp_part;  // proportion of ionized impurity as fraction of n_D
  PetscReal imp_atom;  // atomic number of neutral impurity e.g. 18 Argon
  PetscReal imp_charge;  // charge of ionized impurity e.g. 1 for Ar+
  PetscReal imp_I;  // logarithm of reciprocal of mean excitation energy in units of e rest mass
  PetscReal imp_a;  // logarithm of atomic radius parameter from papers

  int     bc_type;
  UWtype diff_type;

  flux_op Uface;  // a functor providing the algorithm for interpolating the flux
  PetscScalar vi, vip12, vip1;
  PetscScalar xij, xijp12, xijp1;

  std::vector<PetscReal>  aTv, bTv, dTv;

  bool dw_;
  bool partial_;
public:
  fk_equ(mesh *mesh_, char* param_file);

  virtual ~fk_equ()
    {
    };

  virtual void initialize(Field **x, AppCtx *user) = 0;

  /*
   * Eval set the equation
   */
  virtual void Eval(Field **xx, Field **ff, AppCtx *user) = 0;

  /* Set analytic Jacobian for the solver*/
  virtual void SetJacobian(Field **x, Mat jac, AppCtx *user) = 0;

    /*
   * provide an external hook for necessary update of the equations
   */
  virtual void update(Field **xx, AppCtx *user) = 0;

  inline PetscScalar Efield(PetscScalar r)
  {
    return E0; // * 0.5*(1.0 - tanh((r-0.5)/(0.1)));
  };
    
protected:
  virtual PetscScalar Fxi(PetscInt nn, int i, int j, Field **xx) = 0;

  virtual PetscScalar Fv(PetscInt nn, int i, int j, Field **xx) = 0;

  inline PetscScalar Fx(PetscInt nn, int i, int j, Field **xx) {};

  inline PetscScalar Fmax(PetscScalar v)
  {
    return ne * (1.0/(cubic(vte)*(PI*sqrt(PI)))) * PetscExpScalar((1-sqrt(1 + square1(v)))/Dpp);
  };

  inline PetscScalar Delta(PetscScalar v, PetscScalar xi)
  {
    return 1.e-15*PetscExpScalar(-(v-30)*(v-30)/25) * PetscExpScalar(-(xi+0.9)*(xi+0.9)/2.5e-3);
  };
    
  inline PetscReal Chand(PetscReal v)
  {
    PetscReal x = v/sqrt(1.+v*v)/vte;
    return (0.5/(x*x)) * (erf(x) - 2.0*x*PetscExpScalar(-x*x)/sqrt(PI));
  };

  inline PetscReal Lambda_ee(PetscReal gm)
  {
    return (1.0 + beta * log( sqrt(2.*(gm - 1.))/vte));
  };

  inline PetscReal Lambda_ei(PetscReal p)
  {
    return (1.0 + beta * log(2.0*p/vte));
  };

//  inline PetscReal Lambda_ee(PetscReal gm)
//  {
//  return (1.0 + beta * log( sqrt(gm - 1.)));
//  };
//
//  inline PetscReal Lambda_ei(PetscReal p)
//  {
//  return (1.0 + beta * log(sqrt(2.0)*p));
//  };

  inline PetscReal Partial_g(PetscReal p)
  {
    return beta*( (square1(imp_atom) - square1(imp_charge))*(imp_a+log(p)) - TWOTHIRDS*square1(imp_atom - imp_charge));
  };

  inline PetscReal Partial_drag(PetscReal p)
  {
    return beta*(imp_atom - imp_charge) * ( ((1.0+p*p)/(p*p)) * (log( p*sqrt(sqrt(1.0+p*p) - 1.0) ) + imp_I )  - 1.0 )   ;
  };


};



#endif /* FK_EQU_HPP_ */
