/**
 *  @file fk_equ_slab.h
 *
 *  Created on: Nov 19, 2012
 *    Author: zehuag
 */

#ifndef FK_EQU_SLAB_HPP_
#define FK_EQU_SLAB_HPP_

#include <math.h>

#include "../userdata.h"
#include "../fk_equ.h"
#include "Dwp_slab.h"

#include <vector>

class Simulation;

/**
 * fk_equ_slab describes the fokker-plank equation in slab/cylindrical geometry
 */
class fk_equ_slab : public fk_equ
{
  friend class Simulation;
 protected:
  Dwp_slab          Dw;

 public:
  fk_equ_slab(mesh *mesh_, char* param_file);

  ~fk_equ_slab()
    {
    };

  virtual void initialize(Field **x, AppCtx *user);

  /*
   * Eval set the equation
   */
  virtual void Eval(Field **xx, Field **ff, AppCtx *user);

  /* Set analytic Jacobian for the solver*/
  virtual void SetJacobian(Field **x, Mat jac, AppCtx *user);

  virtual void update(Field **xx, AppCtx *user) 
  {
  };

 protected:

  virtual PetscScalar Fxi(PetscInt nn, int i, int j, Field **xx);

  virtual PetscScalar Fv(PetscInt nn, int i, int j, Field **xx);

};



#endif /* FK_EQU_SLAB_HPP_ */
