/**
 *  @file fk_equ_ba.h
 *
 *  Created on: Nov 19, 2012
 *    Author: zehuag
 */

#ifndef FK_EQU_BA_HPP_
#define FK_EQU_BA_HPP_

#include <math.h>

#include "../userdata.h"
#include "../fk_equ.h"
#include "Dwp_ba.h"
#include <vector>
#include <memory>

class Simulation;

/**
 * fk_equ_ba describes the bounce-averaged fokker-plank equation
 */
template <class knockon_type>
class fk_equ_ba : public fk_equ
{
  friend class Simulation;
 private:

  BACtx  ba_ctx; // contains parameter and functions for bounce-average equation
  std::vector<PetscReal> Ftp, Ftp1, Ftp2; // an extra array taking care the flux across the trap-passing boundary

  std::vector<PetscReal> fxi, fp;

  MPI_Comm commy;
  int recv, send;
  int rank1, rank2;

  Dwp_ba  Dw; // bounce-averaged diffusion operator due to waves

  knockon_type knockon;

  void compute_Ftp(Field **xx);

  void Compute_Fv(Field **xx);

  void Compute_Fxi(Field **xx);


 public:
  fk_equ_ba(mesh *mesh_, Field_EQU *E_Field_, char* param_file);

  ~fk_equ_ba()
    {
    };

  virtual void initialize(Field **x, AppCtx *user);

  /*
   * Eval set the equation
   */
  virtual void EvalNStiff(Field **xx, Field **ff, AppCtx *user);

  virtual void EvalStiff(Field **xx, Field **xdot, Field **ff, AppCtx *user);

  virtual void PrepareInt(Field **x_int, PetscBool ComputeJ=PETSC_TRUE)
  {
	PetscPrintf(PETSC_COMM_WORLD,"fk_equ_ba::PrepareInt() is not supported. I will skip\n");
  };

  virtual PetscScalar EvalMaxdt()
  {
	PetscPrintf(PETSC_COMM_WORLD,"fk_equ_ba::EvalMaxdt() is not supported\n");
    abort();
  };

  virtual void get_kon(Field **xx, AppCtx *user)
  {
	PetscPrintf(PETSC_COMM_WORLD,"fk_equ_ba::get_kon() is not supported. I will skip\n");
  };

  /* Set analytic Jacobian for the solver*/
  virtual void SetIJacobian(Field **x, Field **xdot, PetscReal a, Mat jac, AppCtx *user);

  /*
   * provide an external hook for necessary update of the equations
   */
  virtual void update(Field **xx, AppCtx *user) 
  {
    knockon.update(xx, user);
//	if (my_mesh->with_trap)
//		compute_Ftp(xx);

//    double dZ = state.Z;
//    state = E_Field->get_state();
//    dZ = state.Z - dZ;
//    if (dZ != 0)
//    	update_dTv(dZ);
  }

  inline double eps() { return my_mesh->eps;};

 protected:

  virtual PetscScalar Fxi(PetscInt nn, int i, int j, Field **xx, PetscBool implicit)
  {
  	if (j == my_mesh->My - 1)  {
  		return 0.0;
  	} else if (j == -1) {
  		return 0.0;
  	} else if (j == my_mesh->My2-1 && my_mesh->with_trap) {
  		return 0.0;
  	} else {
  		return fxi[(j-my_mesh->ys+1)*my_mesh->xm + i-my_mesh->xs];
  	}

  };

  virtual PetscScalar Fv(PetscInt nn, int i, int j, Field **xx, PetscBool implict)
  {
  	return fp[(j-my_mesh->ys)*(my_mesh->xm+1) + i-my_mesh->xs+1];
  };


};



#endif /* FK_EQU_BA_HPP_ */
