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

#include "userdata.h"
#include "../mesh.h"
#include "diff_type.h"
#include "../field/Field_EQU.h"

#define square1(x) (x*x)
#define cubic(x) (x*x*x)
#define quad(x) (x*x*x*x)

class Simulation;

/**
 * fk_equ is an interface class (with common data and method members) for fokker-plank equation
 */

class fk_equ
{
  friend class Simulation;
 protected:
  mesh      *my_mesh;
  Field_EQU *E_Field;

  PetscReal   Z;   // Charge of ion species
  PetscReal   rho, ctr, E0, sr; // rho/a, c\tau/R_0, E/E_c (initial E0), \tau/\tau_s
  PetscReal   asp, q;  // aspect ratio, safety factor

  double ne, vte, Dpp;  // density, thermal velocity, energy diffusion related coefficient
  PetscReal   p_cutoff; // cutoff p value in runaway current evaluation
  PetscReal   ra;   // radial location
  PetscReal beta;  // one over base Coulomb logrithm
  PetscReal imp_part;  // proportion of ionized impurity as fraction of n_D
  PetscReal imp_atom;  // atomic number of neutral impurity e.g. 18 Argon
  PetscReal imp_charge;  // charge of ionized impurity e.g. 1 for Ar+
  PetscReal imp_I;  // logarithm of reciprocal of mean excitation energy in units of e rest mass
  PetscReal imp_a;  // logarithm of atomic radius parameter from papers

  int     bc_type;  // type of boundary condition at high energy
  UWtype  diff_type;  // type of upwind scheme

  flux_op Uface;  // a functor providing the algorithm for interpolating the fluxes on the volume boundaries
  PetscScalar vi, vip12, vip1;   // internal variables for energy coordinates
  PetscScalar xij, xijp12, xijp1;  // internal variables for pitch-angle coordinates

  bool dw_, ko_; // flags for wave-particle and knockon collision
  bool partial_;

  std::vector<PetscReal>  aTv, bTv, dTv;  // arrays to store collisional coefficients

 public:
  fk_equ(mesh *mesh_, Field_EQU *E_Field_, char* param_file);

  virtual ~fk_equ()
    {
    };

  virtual void initialize(Field **x, AppCtx *user) = 0;

  /*
   * EvalNStiff set the residual for explicit part
   */
  virtual void EvalNStiff(Field **xx, Field **ff, AppCtx *user) = 0;

  /*
   * EvalStiff set the residual for implicit part
   */
  virtual void EvalStiff(Field **xx, Field **xdot, Field **ff, AppCtx *user) = 0;

  /*
   * PrepareInt set the x_int for computing runaway current
   */
  virtual void PrepareInt(Field **x_int, PetscBool ComputeJ=PETSC_TRUE) = 0;

  /* Set analytic Jacobian for the solver*/
  virtual void SetIJacobian(Field **x, Field **xdot, PetscReal a, Mat jac, AppCtx *user) = 0;

  /* compute max dt for IMEX case */
  virtual PetscScalar EvalMaxdt() {};
  /*
   * provide an external hook for necessary update of the equations
   */
  virtual void update(Field **xx, AppCtx *user) = 0; 

  /* Get knock on source */
  virtual void get_kon(Field **xx, AppCtx *user) = 0;

  /*
   * update the pitch-angle collision coefficient using the change in effective charge
   */
  void update_dTv(double &dZ);

 protected:
  virtual PetscScalar Fxi(PetscInt nn, int i, int j, Field **xx, PetscBool implicit) = 0;

  virtual PetscScalar Fv(PetscInt nn, int i, int j, Field **xx, PetscBool implicit) = 0;


  PetscScalar Fx(PetscInt nn, int i, int j, Field **xx) {};

  inline double Fmax(double v)
  {
    return ne * (1.0/(cubic(vte)*(PI*sqrt(PI)))) * exp((1-sqrt(1 + square1(v)))/Dpp);
  };

  inline double Delta(double v, double xi)
  {
      //checkme: this is different from stead-2d
//    return 1.e-3*PetscExpScalar(-(v-0.1)*(v-0.1)/(0.03*0.03)) * PetscExpScalar(-(xi+1.0)*(xi+1.0)/0.2);
   return 1.e-15*exp(-(v-40)*(v-40)/25) * exp(-(xi+0.9)*(xi+0.9)/2.5e-3);
//    return 1.e-15*exp(-(v-80)*(v-80)/25) * exp(-(xi+0.9)*(xi+0.9)/(2.5e-3));
//    return 1.e-15*exp(-(v-30)*(v-30)/25) * exp(-(xi+0.9)*(xi+0.9)/(2.5e-3));
  };

  inline double Chand(double v)
  {
    double x = v/sqrt(1.+v*v)/vte;
    return (0.5/(x*x)) * (erf(x) - 2.0*x*exp(-x*x)/sqrt(PI));
  };

  inline double Lambda_ee(double gm)
  {
    return (1.0 + beta * log( sqrt(2.*(gm - 1.))/vte));
  };

  inline double Lambda_ei(double p)
  {
    return (1.0 + beta * log(2.0*p/vte));
  };

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
