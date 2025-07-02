/*
 * simulation.cpp
 *
 *  Created on: Sep 26, 2014
 *      Author: zehuag
 */

#include "simulation.h"
#include <string.h>             /* strcmp */
#include <sys/stat.h>
#include <stdio.h>
#include <math.h>


//The following external functions are required for PETSC TS solvers

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Setting implicit parts of the equation
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
static PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec x, Vec Xdot, Vec f, void *ctx)
{
	AppCtx           *user = (AppCtx*) ctx;
	DM				da = user->da;

	PetscErrorCode   ierr;

	Vec              xlocal;
	Field            **ff, **xx, **xdot;

	PetscFunctionBegin;

	ierr = DMGetLocalVector(da, &xlocal); CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

	ierr = DMDAVecGetArray(da, xlocal, &xx); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, Xdot, &xdot); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, f, &ff); CHKERRQ(ierr);

	user->udEqu->EvalStiff(xx, xdot, ff, user);

	ierr = DMDAVecRestoreArray(da,f,&ff); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, Xdot, &xdot); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da,xlocal,&xx); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da,&xlocal); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Setting explicit parts of the equation
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec x, Vec f, void *ctx)
{
	AppCtx           *user = (AppCtx*) ctx;
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

	user->udEqu->EvalNStiff(xx, ff, user);

	ierr = DMDAVecRestoreArray(da,f,&ff); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da,xlocal,&xx); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da,&xlocal); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   compute integral of runaway current
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
PetscScalar ComputeRunawayCurrent(TS ts, Vec x_int, void *ctx)
{
	AppCtx          *user = (AppCtx*) ctx;
	DM				da = user->da;
	PetscErrorCode  ierr;
	Vec             xlocal;
	Field           **xx;
    PetscScalar     j_runaway_;

	PetscFunctionBegin;
	ierr = DMGetLocalVector(da, &xlocal); CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da,x_int,INSERT_VALUES,xlocal); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da,x_int,INSERT_VALUES,xlocal); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, xlocal, &xx); CHKERRQ(ierr);

	user->udEqu->PrepareInt(xx);

	ierr = DMDAVecRestoreArray(da,xlocal,&xx); CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(da, xlocal, INSERT_VALUES, x_int);
    ierr = DMLocalToGlobalEnd(da, xlocal, INSERT_VALUES, x_int);
	ierr = DMRestoreLocalVector(da,&xlocal); CHKERRQ(ierr);

    VecSum(x_int, &j_runaway_);
    j_runaway_=-j_runaway_;  //flip the sign of J_RE due to electron's negative charge
	PetscFunctionReturn(j_runaway_);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   compute integral of runaway density
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
PetscScalar ComputeRunawayDensity(TS ts, Vec x_int, void *ctx)
{
	AppCtx          *user = (AppCtx*) ctx;
	DM				da = user->da;
	PetscErrorCode  ierr;
	Vec             xlocal;
	Field           **xx;
    PetscScalar     n_runaway_;

	PetscFunctionBegin;
	ierr = DMGetLocalVector(da, &xlocal); CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da,x_int,INSERT_VALUES,xlocal); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da,x_int,INSERT_VALUES,xlocal); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, xlocal, &xx); CHKERRQ(ierr);

	user->udEqu->PrepareInt(xx, PETSC_FALSE);

	ierr = DMDAVecRestoreArray(da,xlocal,&xx); CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(da, xlocal, INSERT_VALUES, x_int);
    ierr = DMLocalToGlobalEnd(da, xlocal, INSERT_VALUES, x_int);
	ierr = DMRestoreLocalVector(da,&xlocal); CHKERRQ(ierr);

    VecSum(x_int, &n_runaway_);
	PetscFunctionReturn(n_runaway_);
}


/* ---------------------------------------------------------------------
   IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
   a is the a positive shift determined by the method
   ---------------------------------------------------------------------*/
static PetscErrorCode FormIJacobian(TS ts, PetscReal t,Vec x, Vec Xdot, PetscReal a, Mat J, Mat Jpre, void *ctx)
{
	AppCtx    		*user = (AppCtx*) ctx;
	PetscErrorCode 	ierr;
	Vec 			xlocal;
	Field   		**xx, **xdot;
	DM      		da = user->da;

	PetscFunctionBegin;

	DMGetLocalVector(da, &xlocal);
	DMGlobalToLocalBegin(da, x, INSERT_VALUES, xlocal);
	DMGlobalToLocalEnd(da, x, INSERT_VALUES, xlocal);

	/* Get pointers to vector data */
	DMDAVecGetArray(da, xlocal, &xx);
	DMDAVecGetArray(da, Xdot, &xdot);

	user->udEqu->SetIJacobian(xx, xdot, a, Jpre, user);

	/* Restore vectors */
	DMDAVecRestoreArray(da, xlocal, &xx);
	DMDAVecRestoreArray(da, Xdot, &xdot);
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
	ierr = MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


/**
 * A function to monitor the solution
 */
static PetscErrorCode MyMonitor(TS ts, PetscInt step, PetscReal crtime, Vec x, void *ctx)
{
  PetscErrorCode 	ierr;
  PetscReal time;

//  TSGetTimeStepNumber(ts, &step);
  TSGetTime(ts, &time);
  AppCtx    		*user = (AppCtx*) ctx;

  PetscPrintf(PETSC_COMM_WORLD, "Timestep %D: time = %g\n", step, time);
  if ((time - user->dt0) >= user->dt || time == 0. || time >= user->dt1) {
  	PetscViewer   outputfile; /* file to output data to */
  	char filename[50];
  	PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
  	PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
  	PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
  	sprintf(filename, "output_data/soln_%d.dat", step);
  	PetscViewerFileSetName(outputfile, filename);
  
  	VecView(x, outputfile);
  
  	PetscViewerDestroy(&outputfile);
  	PetscPrintf(PETSC_COMM_WORLD, "Data saved at step %D\n", step);
  
  	user->dt0 = time;
  
  	int rank;
  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  	if (rank == 0) {
  		std::ofstream file("output_data/time.txt", std::ios::app);
  		if (file.is_open())
  		{
  			file << "step: "<<step<< " time: "<<time <<"\n";
  			file.close();
  		}
  		else
  			std::cout<< "Unable to open file";
  	}
  }
  return 0;
};


Simulation::Simulation(mesh *mesh_, fk_equ *equ, Field_EQU *E_field_) : udEQU(equ), p_mesh(mesh_), E_field(E_field_), dt(0.01), ftime(1), tprev(0.0), skip(1)
{
  PetscBool flag_numerical_analysis = PETSC_FALSE;
  PetscBool flag_dynamical_analysis = PETSC_FALSE;
  PetscBool flag_runaway = PETSC_FALSE;
  PetscBool flag_kon     = PETSC_FALSE;
  PetscBool flag_df      = PETSC_FALSE;

    matrix_free = PETSC_FALSE;
	matfdcoloring = NULL;
	coloring = PETSC_FALSE;

	user.da = p_mesh->da;
	user.udEqu = equ;

	user.dt0 = 0;
    eta = 0.0;
    mu0 = 1e-3;
    j_para = 0.0;
    j_runaway = 0.0;
    j_runaway_old = 0.0;

	PetscOptionsGetReal(NULL, NULL,"-dt",&dt,NULL);
	PetscOptionsGetReal(NULL, NULL,"-ftime",&ftime,NULL);
	PetscOptionsGetReal(NULL, NULL,"-skip",&skip,NULL);

    PetscOptionsGetBool(NULL, NULL,"-flag_numerical_analysis",&flag_numerical_analysis,NULL);
    PetscOptionsGetBool(NULL, NULL,"-flag_dynamical_analysis",&flag_dynamical_analysis,NULL);
    PetscOptionsGetBool(NULL, NULL,"-flag_runaway",&flag_runaway,NULL);
    PetscOptionsGetBool(NULL, NULL,"-flag_kon",&flag_kon,NULL);
    PetscOptionsGetBool(NULL, NULL,"-flag_df", &flag_df, NULL);
	PetscOptionsGetReal(NULL, NULL,"-eta",&eta,NULL);
	PetscOptionsGetReal(NULL, NULL,"-mu0",&mu0,NULL);
	PetscOptionsGetReal(NULL, NULL,"-j_para",&j_para,NULL);
    if (eta>0.0){
        Emodel = 1;
        flag_runaway = PETSC_TRUE;  //turn on in self consistent model
    }
    else{
        Emodel = 0;
    }

	user.dt1 = ftime;
	user.dt = skip;

    user.flag_numerical_analysis=flag_numerical_analysis;
    user.flag_dynamical_analysis=flag_dynamical_analysis;
    user.flag_runaway=flag_runaway;
    user.flag_kon=flag_kon;
    user.flag_df=flag_df;

	PetscPrintf(PETSC_COMM_WORLD,"-------------------- Solver options ------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"dt = %lg\n", dt);
	PetscPrintf(PETSC_COMM_WORLD,"ftime = %lg\n", ftime);
	PetscPrintf(PETSC_COMM_WORLD,"skip = %lg\n", skip);
    if (flag_runaway) PetscPrintf(PETSC_COMM_WORLD,"-------------------- Runaway current computation is on ------------------\n");

	int rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (rank == 0) {
		mkdir("output_data", S_IRWXU);  // make directory for output
		std::ofstream file("output_data/time.txt", std::ios::trunc);
		if (file.is_open())
		{
			file << "# This file contains the time corresponding to the output time steps (adaptive step).\n";
			file.close();
		}
		else{
			std::cout<< "Unable to open file";
        }

        if(flag_runaway){
            std::ofstream file("output_data/runaway.txt", std::ios::trunc);
		    if (file.is_open())
		    {
		    	file << "# This file contains the time corresponding to the output time steps (adaptive step).\n";
		    	file.close();
		    }
            else{
			    std::cout<< "Unable to open file";
            }
        }
	}

	setup();

	PetscPrintf(PETSC_COMM_WORLD,"--------------------Simulation setup Done---------------------\n");
};

/**
 * Stepping algorithm;
 */
PetscErrorCode Simulation::setup()
{
	PetscErrorCode ierr;
	TSType ts_type;   //type of the time stepping scheme

	char           fname[1024];
	PetscBool      flg;
    PetscMPIInt    size;

	PetscFunctionBegin;

	ierr = DMCreateGlobalVector(user.da, &x);CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(user.da, &x_int);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	ierr = TSCreate(PETSC_COMM_WORLD, &ts); CHKERRQ(ierr);
	ierr = TSSetDM(ts, user.da); CHKERRQ(ierr);
	ierr = TSSetProblemType(ts, TS_NONLINEAR); CHKERRQ(ierr);

	// TSSolve embedded monitor routine
//	ierr = TSMonitorSet(ts, MyMonitor, &user, PETSC_NULL); CHKERRQ(ierr);

	//  Obtain runtime options
	ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

	TSGetType(ts, &ts_type);
	PetscPrintf(PETSC_COMM_WORLD, "Using time stepping algorithm: %s\n", ts_type);
    PetscPrintf(PETSC_COMM_WORLD, "Number of ranks: %d\n", size);

    if(user.flag_numerical_analysis) {
      PetscPrintf(PETSC_COMM_WORLD, "Running mode: Numerical Analysis\n");
    }
    if(user.flag_dynamical_analysis) {
      PetscPrintf(PETSC_COMM_WORLD, "Running mode: Dynamical Analysis\n");
    }

	if ( !strcmp(ts_type, TSPSEUDO) ) {
		//TSSetMaxSteps(ts,1e5);
		TSSetDuration(ts, 1e11, 1e18);
		TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
		TSSetTimeStep(ts,1e-4);
		//TSSetMaxTime(ts,1e12);
		TSPseudoSetTimeStep(ts,TSPseudoTimeStepDefault,0);
		TSSetFromOptions(ts); CHKERRQ(ierr);
	} else {
		PetscInt maxsteps = PetscInt(ftime/dt);
		ierr = TSSetDuration(ts, maxsteps, ftime); CHKERRQ(ierr);
		TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP); // setting to match the final time
	}

	// initial CFL condition
	ierr = TSSetInitialTimeStep(ts, 0.0, dt); CHKERRQ(ierr);

	// initialize the field
	E_field->Initialize();
	pstate state_ = E_field->get_state();
	PetscPrintf(PETSC_COMM_WORLD,"Emodel = %d; (0 - fixed E; 1 - simple self consistent model)\n", Emodel);
	PetscPrintf(PETSC_COMM_WORLD,"============================================\n");
    if (Emodel == 0){
	    PetscPrintf(PETSC_COMM_WORLD,"The external field is fixed: \n");
	    PetscPrintf(PETSC_COMM_WORLD,"E = %lg,\n", state_.E);
    }
    else
    {
        j_runaway = 0.0;
        state_.E = eta*(j_para-j_runaway);
        E_field->set_state(state_);
	    PetscPrintf(PETSC_COMM_WORLD,"The external field is evolving: \n");
	    PetscPrintf(PETSC_COMM_WORLD,"E      = %lg,\n", state_.E);
	    PetscPrintf(PETSC_COMM_WORLD,"eta    = %lg,\n", eta);
	    PetscPrintf(PETSC_COMM_WORLD,"j_para = %lg,\n", j_para);
    }
	PetscPrintf(PETSC_COMM_WORLD,"============================================\n");
    

	// Initial guess or load from file if restart
	PetscOptionsGetString(NULL, NULL, "-file", fname, 1024, &flg);

	if (!flg) {
		Field **ff;
		ierr = DMDAVecGetArray(user.da, x, &ff);CHKERRQ(ierr);
		udEQU->initialize(ff, &user);
		ierr = DMDAVecRestoreArray(user.da, x, &ff);CHKERRQ(ierr);
		PetscPrintf(PETSC_COMM_WORLD,"This is a fresh run, distribution initialized with analytical function. \n");

	} else {
		PetscViewer    viewer;
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname, FILE_MODE_READ, &viewer);
		VecLoad(x,viewer);
		PetscViewerDestroy(&viewer);
		PetscPrintf(PETSC_COMM_WORLD,"This is a restart run, distribution initialized from %s\n", fname);
	}

	ierr = TSSetSolution(ts, x); CHKERRQ(ierr);

	TSSetMaxSNESFailures(ts, -1);

	// setting the equations for TS
	if ((!strcmp(ts_type, TSEULER)) || (!strcmp(ts_type, TSRK)) || (!strcmp(ts_type, TSSSP))) {
		// Explicit time-integration
		ierr = TSSetRHSFunction(ts, PETSC_NULL, FormRHSFunction, &user);  CHKERRQ(ierr);
		std::strcpy(user.ts_type, "explicit");

	} else if ((!strcmp(ts_type, TSBEULER)) || (!strcmp(ts_type, TSARKIMEX))
            || (!strcmp(ts_type, TSTHETA)) || (!strcmp(ts_type, TSBDF))
			|| (!strcmp(ts_type, TSROSW)) || !strcmp(ts_type, TSPSEUDO)) {

        PetscBool flg=PETSC_FALSE;
        if (!strcmp(ts_type, TSARKIMEX))
        {
            TSARKIMEXGetFullyImplicit(ts, &flg);
        }

		// Implicit time-integration
		if ((!strcmp(ts_type, TSARKIMEX) && !flg) || (!strcmp(ts_type, TSROSW))) {
			// IMEX needs to set rhs
			ierr = TSSetRHSFunction(ts, PETSC_NULL, FormRHSFunction, &user);  CHKERRQ(ierr);
		    std::strcpy(user.ts_type, "imex");
		}
        else
        {
		    std::strcpy(user.ts_type, "implicit");
        }

		ierr = TSSetIFunction(ts,PETSC_NULL, FormIFunction, &user);  	CHKERRQ(ierr);

		PetscOptionsGetBool(NULL, NULL, "-snes_mf", &matrix_free,NULL);
		PetscOptionsGetBool(NULL, NULL, "-fdjac", &coloring,NULL);

		if (!matrix_free) {
            if(size==1) {
              DMSetMatType(user.da, MATAIJ);
            } else {
              DMSetMatType(user.da, MATMPIAIJ);
            }
            DMSetMatType(user.da, MATAIJ);
			DMCreateMatrix(user.da, &J);

			if (coloring) {
				SNES snes;
				TSGetSNES(ts,&snes);
				// ISColoring iscoloring;
				// DMCreateColoring(user.da,IS_COLORING_GLOBAL,&iscoloring);
				// MatFDColoringCreate(J,iscoloring,&matfdcoloring);
				// MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormFunction,&user);
				// MatFDColoringSetFromOptions(matfdcoloring);
				// MatFDColoringSetUp(J,iscoloring,matfdcoloring);
				// SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,matfdcoloring);
				// ISColoringDestroy(&iscoloring);
				SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,PETSC_NULL); /* This uses the DMDA to determine the non-zero pattern */
				SNESSetFromOptions(snes);

			} else {
				// user provided jacobian evaluation
				TSSetIJacobian(ts,J,J,FormIJacobian,&user);
				SNES snes;
				TSGetSNES(ts,&snes);
				SNESSetFromOptions(snes);
			}
		}
	}
    PetscPrintf(PETSC_COMM_WORLD,"The time integrator is %s\n",user.ts_type);

    PetscFunctionReturn(0);
};


/**
 * solve() is the solving algorithm;
 */
PetscErrorCode Simulation::solve()
{
	PetscErrorCode  ierr;
	TSConvergedReason  ts_reason;                  /* negative if TS failed to converge*/
	PetscInt             steps;          /* linear and nonlinear iterations for convergence */
	PetscScalar       time = 0., dt;
	Field **xx;
	TSType ts_type;

        // Emil
        PetscScalar *data;
        PetscInt N;
        Vec x_copy,x_zeros,F,Fp;

	PetscFunctionBegin;

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     enters main loop, computes delta-f distribution.
     Exits loop when convergence criteria for solution of field particle
     term is met.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	TSGetType(ts, &ts_type);

    if (!strcmp(user.ts_type, "imex"))
    {
	    PetscScalar dtmax = user.udEqu->EvalMaxdt();
	    PetscPrintf(PETSC_COMM_WORLD,"estimated time step from CFL=0.9 is %e,\n", dtmax);
	    if(!user.flag_numerical_analysis) {
              TSSetTimeStep(ts,dtmax);
        }

        //dtmaxin=dtmax;
        //dtminin=1e-12;


	    //PetscOptionsGetReal(NULL, NULL, "-ts_adapt_dt_max",&dtmaxin,NULL);
	    //PetscOptionsGetReal(NULL, NULL, "-ts_adapt_dt_min",&dtminin,NULL);

        //if (dtmaxin>dtmax)
        //{
	    //    PetscPrintf(PETSC_COMM_WORLD,"input max time step is too large, I will use estimated dtmax instead\n");
        //    dtmaxin = dtmax;
        //}

        //if (dtminin>dtmax)
        //{
	    //    PetscPrintf(PETSC_COMM_WORLD,"input min time step is too large, I will use estimated dtmax instead\n");
        //    dtminin = dtmax;
        //}
        //
        //TSGetAdapt(ts,&adapt);
        //TSAdaptSetStepLimits(adapt,dtminin,dtmaxin);

    }

	if ((!strcmp(ts_type, TSPSEUDO))) {
		PetscInt nits, lits;
		TSSolve(ts, x);
		TSGetSolveTime(ts, &ftime);
		//TSGetStepNumber(ts, &steps);
		TSGetSNESIterations(ts, &nits);
		TSGetKSPIterations(ts,&lits);
		//PetscPrintf(PETSC_COMM_WORLD,"Time integrator took (%D,%D,%D) iterations to reach final time %g\n",steps,nits,lits,(double)ftime);
		Monitor(ts);

	} else {
		while ( time < ftime ) {

			// perform update of (parameters, state quantities) for the equation being solved, assuming linear dependence of coefficients and/or the change is slow
			ierr = DMDAVecGetArray(user.da, x, &xx); CHKERRQ(ierr);
			user.udEqu->update(xx, &user);
			ierr = DMDAVecRestoreArray(user.da, x, &xx); CHKERRQ(ierr);

			Monitor(ts);

			TSGetTime(ts, &dt);

                        if(user.flag_dynamical_analysis) {
                           PetscMPIInt    size;
                           ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
                          if(size!=1) {
                            printf("Can only be ran on one processor! (currently using %d)",size);
                            exit(0);
                          }

                          VecGetSize(x,&N);
                          PetscPrintf(PETSC_COMM_WORLD,"Vector size: %d\n", N);

                          PetscScalar epsi=1e-05;
                          VecDuplicate(x,&x_copy);
                          VecDuplicate(x,&x_zeros);
                          VecZeroEntries(x_zeros);
                          VecDuplicate(x,&F);

                          ierr = FormIFunction(ts, 0, x, x_zeros, F,(void *) &user); 	CHKERRQ(ierr);
                          VecDuplicate(F,&Fp);


                          PetscViewer   outputfile; /* file to output data to */
                          char filename[50];

                          PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
                          PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
                          PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
                          sprintf(filename, "output_data/IJacobian.dat");
                          PetscViewerFileSetName(outputfile, filename);

                          for(PetscInt i=0;i<N;i++) {
                            printf("\rImplicit Jacobian construction: %3d%%",int(100.*(i+1.)/N));
                            VecCopy(x,x_copy);
                            VecSetValues(x_copy,1,&i,&epsi,ADD_VALUES);
                            ierr = FormIFunction(ts, 0, x_copy, x_zeros, Fp,(void *) &user); 	CHKERRQ(ierr);
                            VecAXPY(Fp,-1,F);
                            VecScale(Fp, 1./epsi);

                            VecView(Fp, outputfile);

                            //VecGetArray(Fp,&data);
                            //printf("%3d: %f %f %f\n",i,data[0],data[1],data[2]);
                            //VecRestoreArray(Fp,&data);
                           }

                          PetscViewerDestroy(&outputfile);

                          printf("\n");

                          PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
                          PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
                          PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
                          sprintf(filename, "output_data/RHSJacobian.dat");
                          PetscViewerFileSetName(outputfile, filename);

                          ierr = FormRHSFunction(ts, 0, x, F,(void *) &user); 	CHKERRQ(ierr);

                          for(PetscInt i=0;i<N;i++) {
                            printf("\rExplicit Jacobian construction: %3d%%",int(100.*(i+1.)/N));
                            VecCopy(x,x_copy);
                            VecSetValues(x_copy,1,&i,&epsi,ADD_VALUES);
                            ierr = FormRHSFunction(ts, 0, x_copy, Fp,(void *) &user); 	CHKERRQ(ierr);
                            VecAXPY(Fp,-1,F);
                            VecScale(Fp, 1./epsi);

                            VecView(Fp, outputfile);

                            //VecGetArray(Fp,&data);
                            //printf("%3d: %f %f %f\n",i,data[0],data[1],data[2]);
                            //VecRestoreArray(Fp,&data);
                           }

                          PetscViewerDestroy(&outputfile);




                          printf("\nDone!");

                          exit(0);
                        }
			TSStep(ts);
			TSGetTime(ts, &time);
			dt = time - dt; // time step of the previous step
	        PetscPrintf(PETSC_COMM_WORLD,"previous time step is %e,\n", dt);
		}
	}

	ierr = TSGetTime(ts, &ftime); CHKERRQ(ierr);

	ierr = TSGetTimeStepNumber(ts, &steps); CHKERRQ(ierr);

	ierr = TSGetConvergedReason(ts, &ts_reason); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %d steps\n", TSConvergedReasons[ts_reason], ftime, steps);

    if(user.flag_numerical_analysis) {
      SaveSolution(ts);
    }

    PetscFunctionReturn(0);
};


PetscErrorCode Simulation::Monitor(TS ts)
{
	PetscErrorCode 	ierr;
	PetscReal dt, time;
	PetscInt step;

	PetscFunctionBegin;

	TSGetTimeStepNumber(ts, &step);
	TSGetTime(ts, &time);

	PetscPrintf(PETSC_COMM_WORLD, "Timestep %D: time = %g\n", step, time);

	if ((time - tprev) >= skip || time == 0. || time == ftime) {
		PetscViewer   outputfile; /* file to output data to */
		char filename[50];
		PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
		PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
		PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
		sprintf(filename, "output_data/soln_%d.dat", step);
		PetscViewerFileSetName(outputfile, filename);

		VecView(x, outputfile);

		PetscViewerDestroy(&outputfile);
		PetscPrintf(PETSC_COMM_WORLD, "Data saved at step %D\n", step);

        if (user.flag_kon)
        {
            Vec kon;
		    VecDuplicate(x, &kon);
		    Field **ff;
		    ierr = DMDAVecGetArray(user.da, kon, &ff);CHKERRQ(ierr);
		    udEQU->get_kon(ff, &user);
		    ierr = DMDAVecRestoreArray(user.da, kon, &ff);CHKERRQ(ierr);

		    PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
		    PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
		    PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
		    sprintf(filename, "output_data/kon_%d.dat", step);
		    PetscViewerFileSetName(outputfile, filename);
		    VecView(kon, outputfile);
		    PetscViewerDestroy(&outputfile);
		    PetscPrintf(PETSC_COMM_WORLD, "Knock on source saved at step %D\n", step);
	        VecDestroy(&kon);
        }

		tprev = time;

		int rank;
		MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
		if (rank == 0) {
			std::ofstream file("output_data/time.txt", std::ios::app);
			if (file.is_open())
			{
				file << "step: "<<step<< " time: "<<time <<"\n";
				file.close();
			}
			else
				std::cout<< "Unable to open file";
		}
	}

    //compute total runaway current
	if(user.flag_runaway) {
        VecCopy(x, x_int);
        j_runaway_old = j_runaway;
        j_runaway = ComputeRunawayCurrent(ts, x_int, &user);

        PetscScalar n_runaway;
        if (PETSC_TRUE){
            VecCopy(x, x_int);
            n_runaway = ComputeRunawayDensity(ts, x_int, &user);
	        PetscPrintf(PETSC_COMM_WORLD, "runaway density = %g\n", n_runaway);
        }
	    PetscPrintf(PETSC_COMM_WORLD, "runaway current = %g\n", j_runaway);

        /*
        //sanity check
        PetscScalar total_sum;
        VecSum(x, &total_sum);
	    PetscPrintf(PETSC_COMM_WORLD, "raw sum = %15.10f\n", total_sum);
        */

        //compute E model
	    pstate state_ = E_field->get_state();
        if (Emodel==1 && time>0.0)
        {
            PetscScalar  dt;
            TSGetTimeStep(ts, &dt);
            j_para += - dt*eta/mu0*(j_para - (j_runaway_old+j_runaway)/2.0);
            state_.E = eta*(j_para-j_runaway);
            E_field->set_state(state_);
	        PetscPrintf(PETSC_COMM_WORLD,"Updated E = %lg,\n", state_.E);
        }

        //document runaway current
  	    int rank;
  	    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  	    if (rank == 0) {
  	    	std::ofstream file("output_data/runaway.txt", std::ios::app);
  	    	if (file.is_open())
  	    	{
  	    		file << time <<","<<j_runaway;
                if (PETSC_FALSE) file <<","<<n_runaway;
                if (Emodel==1) file <<","<<j_para<<","<<state_.E;
                file << "\n";
  	    		file.close();
  	    	}
  	    	else{
  	    		std::cout<< "Unable to open file";
            }
  	    }
    }
	PetscFunctionReturn(0);
};

PetscErrorCode Simulation::SaveSolution(TS ts)
{
	PetscErrorCode 	ierr;
	PetscReal dt, time;
	PetscInt step;

	PetscFunctionBegin;

	TSGetTimeStepNumber(ts, &step);
	TSGetTime(ts, &time);

	PetscPrintf(PETSC_COMM_WORLD, "Timestep %D: time = %g\n", step, time);
    PetscViewer   outputfile; /* file to output data to */
    char filename[50];
    PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
    PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
    sprintf(filename, "output_data/soln.dat");
    PetscViewerFileSetName(outputfile, filename);

    VecView(x, outputfile);

    PetscViewerDestroy(&outputfile);
    PetscPrintf(PETSC_COMM_WORLD, "Data saved at step %D\n", step);

	PetscFunctionReturn(0);

};

PetscErrorCode Simulation::cleanup()
{
	PetscErrorCode ierr;
	TSType ts_type;

	PetscFunctionBegin;

	TSGetType(ts, &ts_type);

	if (!matrix_free) {ierr = MatDestroy(&J); CHKERRQ(ierr);}
	if (coloring) {ierr = MatFDColoringDestroy(&matfdcoloring); CHKERRQ(ierr);}

	VecDestroy(&x);
	VecDestroy(&x_int);
	TSDestroy(&ts);

	PetscFunctionReturn(0);
};


