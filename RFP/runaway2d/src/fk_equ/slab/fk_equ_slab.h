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
template <class knockon_type>
class fk_equ_slab : public fk_equ
{
  friend class Simulation;
 protected:
  knockon_type knockon;
  Dwp_slab          Dw;

 public:
  fk_equ_slab(mesh *mesh_, Field_EQU *E_Field_, char* param_file);

  ~fk_equ_slab()
    {
    };

  virtual void initialize(Field **x, AppCtx *user);

  /*
   * Eval set the equation
   */
  virtual void EvalNStiff(Field **xx, Field **ff, AppCtx *user);

  virtual void PrepareInt(Field **x_int, PetscBool ComputeJ=PETSC_TRUE);

  virtual void EvalStiff(Field **xx, Field **xdot, Field **ff, AppCtx *user);

  virtual PetscScalar EvalMaxdt();

  /* Set analytic Jacobian for the solver*/
  virtual void SetIJacobian(Field **x, Field **xdot, PetscReal a, Mat jac, AppCtx *user);

  virtual void update(Field **xx, AppCtx *user) 
  {
    PetscPrintf(PETSC_COMM_WORLD, "Set knock on source\n");
    knockon.update(xx, user);
  };
  
  /* Get knock on source */
  virtual void get_kon(Field **xx, AppCtx *user)
  {
      PetscPrintf(PETSC_COMM_WORLD, "Get knock on source\n");
      knockon.output(xx, user);
  }

 protected:

  virtual PetscScalar Fxi(PetscInt nn, int i, int j, Field **xx, PetscBool implicit);

  virtual PetscScalar Fv(PetscInt nn, int i, int j, Field **xx, PetscBool implicit);

  virtual PetscScalar NstiffFxi(PetscInt nn, int i, int j, Field **xx);

  virtual PetscScalar NstiffFv(PetscInt nn, int i, int j, Field **xx);

};



#endif /* FK_EQU_SLAB_HPP_ */
