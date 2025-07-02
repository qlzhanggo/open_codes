/*
 * simulation.h
 *
 *  Created on: Sep 19, 2014
 *      Author: zehuag
 */

#ifndef SIMULATION_H_
#define SIMULATION_H_

#include <iostream>
#include <fstream>
//using namespace std;

#include <petsc.h>
#include <petscdmda.h>
#include <petscksp.h>

#include <assert.h>

#include "fk_equ/userdata.h"
#include "mesh.h"
#include "fk_equ/fk_equ.h"

/**
 * The Simulation class proves useful interfaces to the user to develop physics model
 */
class Simulation
{
public:
	fk_equ	*udEQU;    		// Aggregation to the user defined field equation (reference)
	mesh	*p_mesh;

	AppCtx   user;
	SNES     snes;                         /* nonlinear solver */
	KSP      ksp;
	Vec      x, r;                         /* x: solution, r: function value */
	Mat 	 J, Jmf;

	PetscReal   atol; // absolute tolerance of solver
	PetscReal   rtol; // relative tolerance of solver
	PetscReal   stol; // change of norm between between nonlinear iteration
	PetscReal   ksprtol; // relative tolerance of linear solver
	PetscReal   kspatol; // absolute tolerance of linear solver
	PetscReal   convtol; // convergence tolerance for iterative solution to field particle term
	PetscInt   maxit;
	PetscInt   kspmaxit;

	PetscBool  matrix_free, coloring;
	MatFDColoring  matfdcoloring;

public:
	Simulation(mesh *mesh_, fk_equ *equ);
	
	~Simulation()
	{
		user.da = NULL;
		user.udEqu = NULL;
	}

	PetscErrorCode cleanup();

	PetscErrorCode solve();

	PetscErrorCode output();

        PetscErrorCode output(int it);

private:
        PetscErrorCode setup();

};



#endif /* SIMULATION_H_ */
