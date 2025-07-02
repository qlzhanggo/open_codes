/*
 * simulation.cpp
 *
 *  Created on: Sep 26, 2014
 *      Author: zehuag
 */

#include "simulation.h"
#include <string.h>             /* strcmp */

//The following external functions are required for PETSC TS solvers

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Setting implicit parts of the equation
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
static PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *ctx)
{
	AppCtx          *user = (AppCtx*) ctx;
	DM				da = user->da;

	PetscErrorCode   ierr;

	Vec              xlocal;
	Field            **ff, **xx;

	PetscFunctionBegin;

	ierr = DMGetLocalVector(da, &xlocal); CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

	ierr = DMDAVecGetArray(da, xlocal, &xx); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, f, &ff); CHKERRQ(ierr);

	//  user->udEqu->update(xx, user);
	user->udEqu->Eval(xx, ff, user);

	ierr = DMDAVecRestoreArray(da,f,&ff); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da,xlocal,&xx); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da,&xlocal); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------------
   IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
   a is the a positive shift determined by the method
   ---------------------------------------------------------------------*/
static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat Jpre, void *ctx)
{
	AppCtx    		*user = (AppCtx*) ctx;
	PetscErrorCode 	ierr;
	Vec 			xlocal;
	Field   		**xx;
	DM      		da = user->da;

	PetscFunctionBegin;

	//  user->udEqu->zeroJacobian(Jpre, user);
	MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);

	DMGetLocalVector(da, &xlocal);
	DMGlobalToLocalBegin(da, x, INSERT_VALUES, xlocal);
	DMGlobalToLocalEnd(da, x, INSERT_VALUES, xlocal);

	/* Get pointers to vector data */
	DMDAVecGetArray(da, xlocal, &xx);

	//	MatZeroEntries(Jpre);
	user->udEqu->SetJacobian(xx, Jpre, user);

	/* Restore vectors */
	DMDAVecRestoreArray(da, xlocal, &xx);
	DMRestoreLocalVector(da, &xlocal);

	MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);

	if (J != Jpre) {
		MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
	}

	/*
    Tell the matrix we will never add a new nonzero location to the
    matrix. If we do, it will generate an error.
	 */
	//	ierr = MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


/* monitor the SNES convergence history for debugging purpose */
PetscErrorCode SNESMonitorError(SNES snes, PetscInt its, PetscReal rnorm, void *ctx)
{
	AppCtx        *user = (AppCtx *) ctx;
	Vec            x;

	PetscViewer       outputfile;                   /* file to output data to */
	char filename[50];

	PetscFunctionBegin;

	PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
	PetscViewerSetType(outputfile, PETSCVIEWERBINARY);

	PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);

	sprintf(filename, "./soln_it_%d.dat", its);

	PetscViewerFileSetName(outputfile, filename);

	SNESGetSolution(snes, &x);
	VecView(x, outputfile);

	PetscViewerDestroy(&outputfile);

	PetscFunctionReturn(0);
}


Simulation::Simulation(mesh *mesh_, fk_equ *equ) : udEQU(equ), p_mesh(mesh_)
{
	user.da = p_mesh->da;
	user.udEqu = equ;

	atol = 1e-12; rtol = 1e-12; stol = 1e-10; ksprtol = 1e-8; kspatol = 1e-12; convtol = 1e-8; maxit = 0;
    kspmaxit=1e4;

	matrix_free = PETSC_FALSE;
	matfdcoloring = NULL;
	coloring = PETSC_FALSE;

	PetscOptionsGetReal(NULL, NULL, "-atol",&atol, NULL);
	PetscOptionsGetReal(NULL, NULL, "-rtol",&rtol,NULL);
	PetscOptionsGetReal(NULL, NULL, "-stol",&stol,NULL);
	PetscOptionsGetReal(NULL, NULL, "-ksprtol",&ksprtol,NULL);
	PetscOptionsGetReal(NULL, NULL, "-kspatol",&kspatol,NULL);
	PetscOptionsGetReal(NULL, NULL, "-convtol",&convtol,NULL);
	PetscOptionsGetInt(NULL, NULL, "-maxit",&maxit,NULL);
	PetscOptionsGetInt(NULL, NULL, "-kspmaxit",&kspmaxit,NULL);

	PetscPrintf(PETSC_COMM_WORLD,"-------------------- Solver options ------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"atol = %lg\n", atol);
	PetscPrintf(PETSC_COMM_WORLD,"rtol = %lg\n", rtol);
	PetscPrintf(PETSC_COMM_WORLD,"stol = %lg\n", stol);
	PetscPrintf(PETSC_COMM_WORLD,"ksprtol = %lg\n", ksprtol);
	PetscPrintf(PETSC_COMM_WORLD,"kspatol = %lg\n", kspatol);
	PetscPrintf(PETSC_COMM_WORLD,"convrtol = %lg\n", convtol);
	PetscPrintf(PETSC_COMM_WORLD,"maxit (number of snes solve) = %d\n", maxit);
	PetscPrintf(PETSC_COMM_WORLD,"kspmaxit = %d\n", kspmaxit);

	setup();
	PetscPrintf(PETSC_COMM_WORLD,"--------------------Simulation setup Done---------------------\n");

}

/**
 * Solver routine
 */

//typedef enum {JACOBIAN_ANALYTIC,JACOBIAN_FD_COLORING,JACOBIAN_FD_FULL} JacobianType;
//static const char *const JacobianTypes[] = {"analytic","fd_coloring","fd_full","JacobianType","fd_",0};

PetscErrorCode Simulation::setup()
{
	PetscErrorCode ierr;
	//	JacobianType   jacType = JACOBIAN_ANALYTIC;  //type of jacobian used for the SNES solver in TS implicit

	char           fname[1024];
	PetscBool      flg = PETSC_FALSE;

	PetscFunctionBegin;

	ierr = DMCreateGlobalVector(user.da, &x);//CHKERRQ(ierr);
	ierr = VecDuplicate(x, &r);//CHKERRQ(ierr);

	// Initial guess
	PetscOptionsGetString(NULL, NULL, "-file", fname, 1024, &flg);

	if (!flg) {
		PetscPrintf(PETSC_COMM_WORLD,"Initialize with Maxwellian distribution.\n");

		Field **xx;
		ierr = DMDAVecGetArray(user.da, x, &xx);CHKERRQ(ierr);
		udEQU->initialize(xx, &user);
		ierr = DMDAVecRestoreArray(user.da, x, &xx);CHKERRQ(ierr);

	} else {
		PetscPrintf(PETSC_COMM_WORLD,"Initialize by reading a file.\n");

		PetscViewer    viewer;
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname, FILE_MODE_READ, &viewer);
		VecLoad(x, viewer);
		PetscViewerDestroy(&viewer);

        PetscBool         checkF = PETSC_FALSE;
        PetscOptionsGetBool(NULL,NULL,"-checkF",&checkF,NULL);
        if (checkF)
        {
	        Vec              xlocal;
	        Field            **ff, **xx;
            PetscReal        residual;

	        ierr = DMGetLocalVector(user.da, &xlocal); CHKERRQ(ierr);
	        ierr = DMGlobalToLocalBegin(user.da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
	        ierr = DMGlobalToLocalEnd(user.da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

	        ierr = DMDAVecGetArray(user.da, xlocal, &xx); CHKERRQ(ierr);
	        ierr = DMDAVecGetArray(user.da, r, &ff); CHKERRQ(ierr);

	        udEQU->Eval(xx, ff, &user);

	        ierr = DMDAVecRestoreArray(user.da,r,&ff); CHKERRQ(ierr);
	        ierr = DMDAVecRestoreArray(user.da,xlocal,&xx); CHKERRQ(ierr);
	        ierr = DMRestoreLocalVector(user.da,&xlocal); CHKERRQ(ierr);

            VecNorm(r,NORM_2,&residual);
            PetscPrintf(PETSC_COMM_WORLD,"\n====== Read-in solution resiudal = %g ======\n\n",residual);
        }

	}

	/*-------------------------------------------------------------
    sets number of non-zero matrix entries. Significantly reduces
    memory usage for cases with large DOF.
    --------------------------------------------------------------*/
	//ierr = MatMPIAIJSetPreallocation(J, 21, NULL, 42, NULL);  //CHKERRQ(ierr);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local function evaluation routine, tolerances, etc.
     A large number of function evaluations is needed if the analytic
     jacobian is not used for solving the linearized Fokker-Planck equation below.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	ierr = SNESCreate(PETSC_COMM_WORLD, &snes); //CHKERRQ(ierr);
	ierr = SNESSetDM(snes, user.da); //CHKERRQ(ierr);

	//      ierr = DMCreateGlobalVector(da, &x);
	//      VecDuplicate(x, &b);

	ierr = SNESSetFunction(snes, r, FormFunction, &user);//CHKERRQ(ierr);

	SNESLineSearch    linesearch;
	char ls_type[16];
	PetscBool ls_flg=PETSC_FALSE;

	PetscPrintf(PETSC_COMM_WORLD, "setting lineaserch type\n");

	//PetscOptionsHasName(NULL,NULL, "-snes_linesearch_type",&ls_flg);
	SNESGetLineSearch(snes, &linesearch);
	PetscOptionsGetString(NULL, NULL, "-snes_linesearch_type", ls_type, sizeof(ls_type), &ls_flg);

	if (ls_flg & ls_type != NULL) {
		if (std::strcmp(ls_type, "cp")==0 || std::strcmp(ls_type, "basic")==0 || std::strcmp(ls_type, "nleqerr")==0 || std::strcmp(ls_type, "l2")==0) {
			PetscPrintf(PETSC_COMM_WORLD, "Using lineaserch type %s\n", ls_type);
			SNESLineSearchSetType(linesearch, ls_type);
			SNESSetLineSearch(snes, linesearch);
			//			SNESLineSearchSetFromOptions(linesearch);
		} else {
			PetscPrintf(PETSC_COMM_WORLD, "Using lineaserch type bt\n");
			SNESLineSearchSetType(linesearch, "bt");
		}
	} else {
		PetscPrintf(PETSC_COMM_WORLD, "Using lineaserch type bt\n");
		SNESLineSearchSetType(linesearch, "bt");
		SNESSetLineSearch(snes, linesearch);
	}

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine.
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning (unless user explicitly sets preconditioner)
     -snes_mf_operator : form preconditioning matrix as set by the user, but use matrix-free approx for Jacobian-vector products within Newton-Krylov method
     -fdcoloring : using finite differences with coloring to compute the Jacobian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	//PetscOptionsGetBool(NULL, NULL, "-snes_mf", &matrix_free, NULL);
	PetscOptionsGetBool(NULL, NULL, "-fdjac", &coloring, NULL);

	J = NULL; Jmf = NULL;

	//if (!matrix_free)  // this snes_mf really should not be added here -QT
    
	// J will be used as preconditioner if using -snes_mf_operator

	DMSetMatType(user.da, MATAIJ);
	DMCreateMatrix(user.da, &J);

	MatCreateSNESMF(snes,&Jmf);
	if (coloring) {
		//			ISColoring iscoloring;
		//			DMCreateColoring(user.da,IS_COLORING_GLOBAL, &iscoloring);
		//			MatFDColoringCreate(J, iscoloring, &matfdcoloring);
		////			MatFDColoringUseDM(J,matfdcoloring);  // use this when IS_COLORING_LOCAL is used
		//			MatFDColoringSetFunction(matfdcoloring, (PetscErrorCode (*)(void))FormFunction, &user);
		//
		//			MatFDColoringSetFromOptions(matfdcoloring);
		//			MatFDColoringSetUp(J,iscoloring,matfdcoloring);
		//
		//			SNESSetJacobian(snes, Jmf, J, SNESComputeJacobianDefaultColor, matfdcoloring);
		//			ISColoringDestroy(&iscoloring);
		SNESSetJacobian(snes,Jmf,J,SNESComputeJacobianDefaultColor,PETSC_NULL);

	} else {
		// user provided jacobian evaluation
		SNESSetJacobian(snes,J, J, FormJacobian, &user);

		// SNESComputeJacobian(snes, x, J, J);
		// PetscViewer viewer;
		// PetscViewerASCIIOpen(PETSC_COMM_WORLD, "jacmat.m", &viewer);
		// PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
		// MatView(J, viewer);
		// PetscViewerDestroy(&viewer);
	}

	ierr = SNESSetFromOptions(snes);//CHKERRQ(ierr);

	//  SNESMonitorSet(snes, SNESMonitorError, &user, NULL); // set monitor for debugging

	PetscFunctionReturn(0);
};


/**
 * solve() is the solving algorithm;
 */
PetscErrorCode Simulation::solve()
{
	PetscErrorCode  ierr;
	SNESConvergedReason  snes_reason;                  /* negative if SNES failed to converge*/

	PetscFunctionBegin;

	int counter = 0;
	bool converged = false;
	Vec  xdiff, xold;
	PetscReal norm_x, norm_xold, norm_xdiff;
	Field **xx;


	if (maxit > 0) {
		ierr = VecDuplicate(x, &xold);
		ierr = VecDuplicate(x, &xdiff);

		ierr = SNESSetTolerances(snes, 1.0e-10, 1.0e-10, 1.0e-10, 200, 5000); // lower the convergence for the first iteration
		ierr = SNESGetKSP(snes, &ksp); // CHKERRQ(ierr);
		ierr = KSPSetTolerances(ksp, 1.e-11, 1.e-11, PETSC_DEFAULT, kspmaxit);//CHKERRQ(ierr);

		while (!converged) {
			// treat the avalanche term perturbatively, so it does not contribute to the Jacobian
			ierr = DMDAVecGetArray(user.da, x, &xx); CHKERRQ(ierr);
			udEQU->update(xx, &user);
			ierr = DMDAVecRestoreArray(user.da, x, &xx); CHKERRQ(ierr);

			ierr = SNESSolve(snes, PETSC_NULL, x);  CHKERRQ(ierr);
			ierr = SNESGetConvergedReason(snes, &snes_reason);CHKERRQ(ierr);

			if (snes_reason < 0) {
				ierr = PetscPrintf(PETSC_COMM_WORLD, "SNES failed! Reason %d\n", snes_reason);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"-----------------------------------------\n");   CHKERRQ(ierr);
			} else {
				ierr = PetscPrintf(PETSC_COMM_WORLD, "SNES converged. Reason %d\n", snes_reason);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"-----------------------------------------\n");   CHKERRQ(ierr);
			}

			// computes norms
			ierr = VecWAXPY(xdiff, -1.0, xold, x);		CHKERRQ(ierr);
			ierr = VecNorm(x, NORM_1, &norm_x);				CHKERRQ(ierr);
			ierr = VecNorm(xold, NORM_1, &norm_xold); 	CHKERRQ(ierr);
			ierr = VecNorm(xdiff, NORM_1, &norm_xdiff);		    CHKERRQ(ierr);

			counter += 1;
			//checks to see if convergence criteria are met
			norm_xdiff /= (norm_xold + norm_x);
			if (norm_xdiff < convtol) {
				ierr = PetscPrintf(PETSC_COMM_WORLD,"=========================================\n");   CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"Solution converged after %d iterations\n", counter);   CHKERRQ(ierr);
				converged = true;
			} else if (counter > maxit) {
				ierr = PetscPrintf(PETSC_COMM_WORLD,"=========================================\n");   CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"Solution failed to converge after %d iterations\n", maxit);   CHKERRQ(ierr);
				converged = true;
			} else
				converged = false;

			PetscPrintf(PETSC_COMM_WORLD, "norm_xdiff = %g, norm_xold = %g, norm_x = %g, iteration = %d \n", norm_xdiff, norm_xold, norm_x, counter);

			//saves solution for next iteration
			ierr = VecCopy(x, xold);CHKERRQ(ierr);

			ierr = SNESSetTolerances(snes, atol, rtol, stol, 200, 5000); CHKERRQ(ierr);
			ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
			ierr = KSPSetTolerances(ksp, ksprtol, kspatol, PETSC_DEFAULT, kspmaxit); CHKERRQ(ierr);
			//			}
			//      output(counter);
			ierr = KSPSetFromOptions(ksp);
			ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
		}

		ierr = VecDestroy(&xdiff); CHKERRQ(ierr);
		ierr = VecDestroy(&xold); CHKERRQ(ierr);

	} else {
		ierr = SNESSetTolerances(snes, atol, rtol, stol, 50, 100);//CHKERRQ(ierr);

		ierr = SNESGetKSP(snes, &ksp); // CHKERRQ(ierr);
		ierr = KSPSetTolerances(ksp, ksprtol, kspatol, PETSC_DEFAULT, kspmaxit);//CHKERRQ(ierr);
		ierr = KSPSetFromOptions(ksp);
		//ierr = KSPSetUp(ksp);

		ierr = SNESSetFromOptions(snes);//CHKERRQ(ierr);

		ierr = SNESSolve(snes, PETSC_NULL, x);  CHKERRQ(ierr);

//		SNESGetJacobian(snes, NULL, &J, NULL, NULL);

		//SNESComputeJacobian(snes, x, Jmf, J);
//		PetscViewer viewer;
//		PetscViewerASCIIOpen(PETSC_COMM_WORLD, "Amat.m", &viewer);
//		PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
//		MatView(J, viewer);
//		PetscViewerPopFormat(viewer);
//		PetscViewerDestroy(&viewer);

//		char filename[50];
//		PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
//		PetscViewerSetType(viewer, PETSCVIEWERBINARY);
//		PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
//		sprintf(filename, "./Amat.dat");
//		PetscViewerFileSetName(viewer, filename);
//		MatView(J, viewer);
//		PetscViewerDestroy(&viewer);

		//MatView(J, PETSC_VIEWER_STDOUT_WORLD);
		ierr = SNESGetConvergedReason(snes, &snes_reason);CHKERRQ(ierr);

		if (snes_reason < 0) {
			ierr = PetscPrintf(PETSC_COMM_WORLD, "SNES failed! Reason %d\n", snes_reason);CHKERRQ(ierr);
		} else {
			ierr = PetscPrintf(PETSC_COMM_WORLD, "SNES converged. Reason %d\n", snes_reason);CHKERRQ(ierr);
		}


	}

	output();

	PetscFunctionReturn(0);
};


#include <stdio.h>

PetscErrorCode Simulation::output()
{
	PetscErrorCode         ierr;
	PetscViewer            outputfile;                   /* file to output data to */
	char filename[50];

	PetscFunctionBegin;

	PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
	PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
	PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
	sprintf(filename, "./soln.dat");
	PetscViewerFileSetName(outputfile, filename);
	VecView(x, outputfile);
	PetscViewerDestroy(&outputfile);

	PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
	PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
	PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
	sprintf(filename, "./residual.dat");
	PetscViewerFileSetName(outputfile, filename);
	VecView(r, outputfile);
	PetscViewerDestroy(&outputfile);

	PetscFunctionReturn(0);

};

PetscErrorCode Simulation::output(int count)
{
	PetscErrorCode         ierr;
	PetscViewer            outputfile;                   /* file to output data to */
	char filename[50];

	PetscFunctionBegin;

	PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
	PetscViewerSetType(outputfile, PETSCVIEWERBINARY);

	PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
	sprintf(filename, "./soln_%d.dat", count);
	PetscViewerFileSetName(outputfile, filename);

	VecView(x, outputfile);

	PetscViewerDestroy(&outputfile);

	PetscFunctionReturn(0);

};

PetscErrorCode Simulation::cleanup()
{
	PetscErrorCode ierr;

	ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = VecDestroy(&r); CHKERRQ(ierr);
	if (!matrix_free) {
		ierr = MatDestroy(&J); CHKERRQ(ierr);
		ierr = MatDestroy(&Jmf); CHKERRQ(ierr);
	}
	if (coloring)
		ierr = MatFDColoringDestroy(&matfdcoloring); CHKERRQ(ierr);

	ierr = SNESDestroy(&snes); CHKERRQ(ierr);

	PetscFunctionReturn(0);
};

