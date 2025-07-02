
/*
 * fk_equ.cpp
 *
 *  Created on: Sep 25, 2014
 *      Author: zehuag
 */

#include "fk_equ.h"


/* constructor
 * reading parameters and set up the collision operator
 * */
fk_equ::fk_equ(mesh *mesh_, char* param_file) : my_mesh(mesh_), Z(1.0), rho(0.03), ctr(10.0), E0(1.0), sr(0.01429), 
              asp(0.3), q(1.0), ra(0.5), beta(0.0667), bc_type(0), diff_type(QUICK),
              ne(1.0), vte(0.01), Dpp(0.5e-4), dw_(false),
			  imp_part(0), partial_(false), imp_atom(18.0), imp_charge(1.0), imp_I(7.8), imp_a(4.5)
{
    int i, j, nn;
	PetscReal v;

	char  foo[30];


	FILE *pfile = fopen(param_file, "r");

	if (pfile == NULL) {
		printf("Unable to open file %s\n", param_file);
		exit(1);
	}

	int diff_t;

	fscanf(pfile, "%s = %lg;\n" ,foo, &Z);
	fscanf(pfile, "%s = %lg;\n" ,foo, &E0);
	fscanf(pfile, "%s = %lg;\n" ,foo, &vte);
	fscanf(pfile, "%s = %lg;\n" ,foo, &sr);
	fscanf(pfile, "%s = %lg;\n" ,foo, &rho);
	fscanf(pfile, "%s = %lg;\n" ,foo, &ctr);
	fscanf(pfile, "%s = %lg;\n" ,foo, &asp);
	fscanf(pfile, "%s = %lg;\n" ,foo, &q);
	fscanf(pfile, "%s = %lg;\n" ,foo, &ra);
	fscanf(pfile, "%s = %lg;\n" ,foo, &beta);
	fscanf(pfile, "%s = %i;\n", foo, &bc_type);
	fscanf(pfile, "%s = %i;\n", foo, &diff_t);
	fgets(foo, 30, pfile);
	fgets(foo, 30, pfile);
	fgets(foo, 30, pfile);
	fgets(foo, 30, pfile);
	fgets(foo, 30, pfile);
	fscanf(pfile, "%s = %lg;\n", foo, &imp_part);
	fscanf(pfile, "%s = %lg;\n", foo, &imp_atom);
	fscanf(pfile, "%s = %lg;\n", foo, &imp_charge);
	fscanf(pfile, "%s = %lg;\n", foo, &imp_I);
	fscanf(pfile, "%s = %lg;\n", foo, &imp_a);

	fclose(pfile);

	if (diff_t>=0 && diff_t<7)
		diff_type = static_cast<UWtype>(diff_t);
	else
		PetscPrintf(PETSC_COMM_WORLD,"Wrong diff_type, using default value: QUICK . \n");

	PetscPrintf(PETSC_COMM_WORLD,"-------------------- Equation parameters ------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"Z = %lg\n", Z);
	PetscPrintf(PETSC_COMM_WORLD,"E0 = %lg\n", E0);
	PetscPrintf(PETSC_COMM_WORLD,"vte = %lg\n", vte);
	PetscPrintf(PETSC_COMM_WORLD,"tau/tau_s = %lg\n", sr);
	PetscPrintf(PETSC_COMM_WORLD,"rho = %lg\n", (rho));
	PetscPrintf(PETSC_COMM_WORLD,"ctr = %lg\n",(ctr));
	PetscPrintf(PETSC_COMM_WORLD,"asp = %lg\n", (asp));
	PetscPrintf(PETSC_COMM_WORLD,"q = %lg\n", (q));
	PetscPrintf(PETSC_COMM_WORLD,"r = %lg\n", (ra));
	PetscPrintf(PETSC_COMM_WORLD,"beta = %lg\n", (beta));
	PetscPrintf(PETSC_COMM_WORLD,"diff_type = %i\n", diff_type);
	PetscPrintf(PETSC_COMM_WORLD,"------------------------------------------------------------\n");

	if (imp_part>0.0){
		partial_ = true;
		PetscPrintf(PETSC_COMM_WORLD,"---------------- Ionized impurity parameters --------------\n");
		PetscPrintf(PETSC_COMM_WORLD,"impurity_frac = %lg\n", imp_part);
		PetscPrintf(PETSC_COMM_WORLD,"impurity_atom = %lg\n", imp_atom);
		PetscPrintf(PETSC_COMM_WORLD,"impurity_charge = %lg\n", imp_charge);
		PetscPrintf(PETSC_COMM_WORLD,"impurity_I = %lg\n", imp_I);
		PetscPrintf(PETSC_COMM_WORLD,"impurity_a = %lg\n", imp_a);
		PetscPrintf(PETSC_COMM_WORLD,"-----------------------------------------------------------\n");
	}


	// Dpp needs to be calculated according to the vte in case vte is set from options
	Dpp = 0.5*vte*vte;

	// setting up collision operator
	aTv.resize(my_mesh->xm+1); bTv.resize(my_mesh->xm+1);
	PetscScalar lambda;
	for (i=my_mesh->xs-1; i<my_mesh->xs+my_mesh->xm; i++) {
		// cell upper face
		if (i == -1 )
			vip1 = my_mesh->xmin;
		else if (i<(my_mesh->Mx))
			vip1 = my_mesh->xf[i];

		lambda = Lambda_ee(sqrt(1. + vip1*vip1));
		aTv[i-my_mesh->xs+1] = sqrt(1.+vip1*vip1)/vip1*Chand(vip1) * lambda;
		bTv[i-my_mesh->xs+1] = Chand(vip1)/Dpp * lambda;
	    if (partial_){
	    	bTv[i-my_mesh->xs+1] += (imp_part/(1+imp_part))*Partial_drag(vip1);
		}


	}

	dTv.resize(my_mesh->xm);
	for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) {
		vip12 = my_mesh->xx[i];
		dTv[i-my_mesh->xs] = 0.5*sqrt(1+square1(vip12))/cubic(vip12) * ( Z * Lambda_ei(vip12)
		+ (erf(vip12/sqrt(1+vip12*vip12)/vte) - Chand(vip12) + Dpp*square1(vip12)/(1.0+square1(vip12)))*Lambda_ee(sqrt(vip12*vip12+1.)) );

		// Add partial screening of a single ion into the scattering operators
		if (partial_){
			dTv[i-my_mesh->xs]+= (0.5*sqrt(1.0+square1(vip12))/cubic(vip12))*( (imp_part/(1+imp_part))*Partial_g(vip12));
		}

	}


};

