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

//#include <petscksp.h>

#include <assert.h>

#include "fk_equ/userdata.h"
#include "mesh.h"
#include "fk_equ/fk_equ.h"
#include "field/Field_EQU.h"




static PetscErrorCode RegisterNewARKMethods(void)
{
  {
    const PetscScalar
      A[4][4] = {{0,0,0,0},
                 {0.5,0,0,0},
		 {0.25,0.25,0,0},
		 {0.25,0.25,0.5,0}},
      b[4]  = {1./4.,1./4.,1./4.,1./4.},
      bt[4] = {1./4.,1./4.,1./4.,1./4.},
      At[4][4] = {{0,0,0,0},
                  {0,0,0,0},
                  {0,0,0,0},
                  {1.0,1.0,1.0,1.0}},
      *bembedt = NULL,*bembed = NULL;
    TSARKIMEXRegister("mprk2fine",2,4,&At[0][0],&bt[0],NULL,&A[0][0],&b[0],NULL,bembedt,bembed,0,NULL,NULL);
  }

  {
    const PetscScalar
      A[1][1] = {{0}},
      b[1]={1.},bt[1]={1.},
      At[1][1] = {{1.0}},
      *bembedt = NULL,*bembed = NULL;
    TSARKIMEXRegister("febe",1,1,&At[0][0],&bt[0],NULL,&A[0][0],&b[0],NULL,bembedt,bembed,0,NULL,NULL);
  }

  {
    const PetscScalar
      A[2][2] = {{0.,0.},{1.,0.}},
      b[2]={0.75,0.25},bt[2]={0.75,0.25},
      At[2][2] = {{0.,0.},{3.,1.}},
      *bembedt = NULL,*bembed = NULL;
    TSARKIMEXRegister("febe2",1,2,&At[0][0],&bt[0],NULL,&A[0][0],&b[0],NULL,bembedt,bembed,0,NULL,NULL);
  }
  
  return(0);
}



/**
 * The Simulation class proves useful interfaces to the user to develop physics model
 */
class Simulation
{
public:
	fk_equ	*udEQU;    		// Aggregation to the user defined field equation (reference)
	mesh	*p_mesh;
	Field_EQU *E_field;

	AppCtx   user;
	TS                     ts;                         /* nonlinear solver */
	Vec                    x, x_int;                   /* x: solution, x_int: temporary vec for integration */
	Mat 				   J;

	PetscScalar   dt, ftime, tprev;
	PetscScalar    skip;
    PetscScalar   j_runaway, j_runaway_old, j_para, eta, mu0;

	PetscBool  matrix_free, coloring;
    PetscInt   Emodel;
	MatFDColoring  matfdcoloring;

public:
	Simulation(mesh *mesh_, fk_equ *equ, Field_EQU *E_field_);

	~Simulation()
	{
	  user.da = NULL;
	  user.udEqu = NULL;
	};

	PetscErrorCode solve();

	PetscErrorCode cleanup();

	PetscErrorCode Monitor(TS ts);

    PetscErrorCode SaveSolution(TS ts);

private:
	PetscErrorCode setup();

};



#endif /* SIMULATION_H_ */
