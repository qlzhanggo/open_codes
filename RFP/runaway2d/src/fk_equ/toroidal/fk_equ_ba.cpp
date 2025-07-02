/*
 * fk_equ_ba.cpp
 *
 *  Created on: Sep 25, 2014
 *      Author: zehuag
 */

#include "fk_equ_ba.h"
#include "knockon_rp_ba.h"
#include "knockon_chiu_ba.h"
#include "knockon_none_ba.h"

/* constructor
 * reading parameters and set up the collision operator
 */
template <class knockon_type>
fk_equ_ba<knockon_type>::fk_equ_ba (mesh *mesh_, Field_EQU *E_Field_, char* param_file) :
fk_equ(mesh_, E_Field_, param_file),
ba_ctx(mesh_), Dw(mesh_, ba_ctx),
knockon(mesh_, &ba_ctx, fk_equ::beta, mesh_->xmin)
{
	int i, j, nn;
	PetscReal v;

	fp.resize((my_mesh->xm+1)*my_mesh->ym, 0.);
	fxi.resize((my_mesh->ym+1)*my_mesh->xm, 0.);

	if (my_mesh->with_trap) {
		// setting the MPI communicator in xi direction
		Ftp1.resize(my_mesh->xm + 2, 0); // include the ghost points \pm 1
		Ftp2.resize(my_mesh->xm + 2, 0); // include the ghost points \pm 1
		Ftp.resize(my_mesh->xm+2, 0);

		int color = (int)(my_mesh->xs/my_mesh->xm);
		int key   = (int)(my_mesh->ys/my_mesh->ym);

		MPI_Comm_split(PETSC_COMM_WORLD, color, key, &commy); //communicator in xi direction

		int rank, size;
		MPI_Comm_rank(commy, &rank);
		MPI_Comm_size(commy, &size);

		int temp(0);

		if (my_mesh->My1>=my_mesh->ys && my_mesh->My1<(my_mesh->ys+my_mesh->ym)) // domain with trap-passing boundary 1{
			temp = rank;
		MPI_Allreduce(&temp, &rank1, 1, MPI_INT, MPI_SUM, commy);

		temp = 0;
		if (my_mesh->My2>=my_mesh->ys && my_mesh->My2<(my_mesh->ys+my_mesh->ym)) // domain with the first point of second passing region
			temp = rank;
		MPI_Allreduce(&temp, &rank2, 1, MPI_INT, MPI_SUM, commy);

		PetscPrintf(PETSC_COMM_WORLD,"rank1 = %d\n", rank1);
		PetscPrintf(PETSC_COMM_WORLD,"rank2 = %d\n", rank2);

		if (rank1 != rank2) {

			if (rank == rank1) {
				send = my_mesh->xm+2;
				MPI_Send(&send, 1, MPI_INT, rank2, 0, commy);
			}
			if (rank == rank2) {
				MPI_Recv(&recv, 1, MPI_INT, rank1, 0, commy, MPI_STATUS_IGNORE);
			}

			if (rank == rank2) {
				send = my_mesh->xm+2;
				MPI_Send(&send, 1, MPI_INT, rank1, 0, commy);
			}
			if (rank == rank1) {
				MPI_Recv(&recv, 1, MPI_INT, rank2, 0, commy, MPI_STATUS_IGNORE);
			}
		}
	}

	double w0, k10, k20, dk10, wpe;
	char   foo[50];
	FILE *pfile = fopen(param_file, "r");

	if (pfile == NULL) {
		printf("Unable to open file %s\n", param_file);
		exit(1);
	}

	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);
	fgets(foo, 50, pfile);

	fscanf(pfile, "%s = %lg;\n" ,foo, &w0);
	fscanf(pfile, "%s = %lg;\n" ,foo, &k10);
	fscanf(pfile, "%s = %lg;\n" ,foo, &k20);
	fscanf(pfile, "%s = %lg;\n" ,foo, &dk10);
	fscanf(pfile, "%s = %lg;\n", foo, &wpe);

	fclose(pfile);


	if (w0>0) {
		Dw.setup(w0, k10, k20, dk10, wpe);
		dw_ = true;

		PetscPrintf(PETSC_COMM_WORLD,"-------------------- wave parameters ------------------\n");
		PetscPrintf(PETSC_COMM_WORLD,"w0 = %lg\n", w0);
		PetscPrintf(PETSC_COMM_WORLD,"k10 = %lg\n", k10);
		PetscPrintf(PETSC_COMM_WORLD,"k20 = %lg\n", k20);
		PetscPrintf(PETSC_COMM_WORLD,"dk10 = %lg\n", dk10);
		PetscPrintf(PETSC_COMM_WORLD,"wpe = %lg\n", (wpe));
		PetscPrintf(PETSC_COMM_WORLD,"------------------------------------------------------------\n");
	}

	PetscPrintf(PETSC_COMM_WORLD, "Runing 2D RFK equation in toroidal geometry with bounce average.\n");
};


/* An iterative method usually requires an initial guess for the function to be solved*/
template <class knockon_type>
void fk_equ_ba<knockon_type>::initialize(Field **xx, AppCtx *user)
{
	int i, j, k, nn;
	PetscReal v, xi;

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi
		xi = my_mesh->yy[j];

		for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { // v

			v = my_mesh->xx[i];

			xx[j][i].fn[0] = ra * Fmax(v)*v + ra * Delta(v,xi)*v;
		}
	}
};


/*
 * The equation for IMEX is assumed to be F(X, Xdot) = RHS(X)
 * EvalStiff set the stiff part (fast) as implicit

 * following is the way our index works
 * coordinates (at cell boundary)               function values (at cell center)
 *                   j+3/2                                       j+1
 *    i-1/2      i+1/2,j+1/2       i+3/2     ----->     i-1     i, j     i+1
 *                   j-1/2                                       j-1
 */
template <class knockon_type>
void fk_equ_ba<knockon_type>::EvalStiff(Field **xx, Field **xdot, Field **ff, AppCtx *user)
{
	PetscInt i, j, k;
	PetscInt   nn = 0;
	PetscReal   aTv_,bTv_,cTv_, dTv_, df, flux;

	Compute_Fxi(xx);
	Compute_Fv(xx);

	if (my_mesh->with_trap)
		compute_Ftp(xx);

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		if (j >= my_mesh->My2)
			xij = my_mesh->yf[j]; // cell lower face
		else
			xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);

		xijp12 = my_mesh->yy[j];  // cell center

		if (j < my_mesh->My2)
			xijp1 = my_mesh->yf[j];   // cell upper face
		else
			xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j+1]) : (1.0);

		for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
			// velocities evaluated at j, at j+1 and j+1/2.
			vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
			vip12 = my_mesh->xx[i];
			vip1 =  my_mesh->xf[i];

			// time derivative
			ff[j][i].fn[nn] = xdot[j][i].fn[nn];

			// energy flux
			ff[j][i].fn[nn] += (Fv(nn, i, j, xx, PETSC_TRUE) - Fv(nn, i-1, j, xx, PETSC_TRUE))/vip12 /(vip1-vi);  //solving v*f

			// pitch-angle flux, it needs special attention at the trap-passing bounary
			if (j == my_mesh->My2) { // trap->passing (+xitp plus)
                
				ff[j][i].fn[nn] += (Fxi(nn, i, j, xx, PETSC_TRUE) - Ftp2[i-my_mesh->xs+1]) /(xijp1-xij)/ba_ctx.zeta1_jp12[j];

			} else if (j == my_mesh->My1+1) { // passing->trap (-xitp plus), 
                
				ff[j][i].fn[nn] += (Fxi(nn, i, j, xx, PETSC_TRUE) - Ftp1[i-my_mesh->xs+1]) /(xijp1-xij)/ba_ctx.zeta1_jp12[j];

			} else if (j == my_mesh->My1) { // passing->trap  (-xitp minus)
                
                //the flux from pass to trap is determined by the conservation of the fluxes
                //i.e.: Flux(+xitp plus)-Flux(-xitp minus) = -2*Flux(-xitp plus)
				ff[j][i].fn[nn] += (2*Ftp1[i-my_mesh->xs+1] + Ftp2[i-my_mesh->xs+1] - Fxi(nn, i, j-1, xx, PETSC_TRUE)) /(xijp1-xij)/ba_ctx.zeta1_jp12[j];

			} else {

				ff[j][i].fn[nn] += (Fxi(nn, i, j, xx, PETSC_TRUE) - Fxi(nn, i, j-1, xx, PETSC_TRUE)) /(xijp1-xij)/ba_ctx.zeta1_jp12[j];
			}

			// add the knockon source term
			ff[j][i].fn[nn] -= ( (1+imp_part*imp_atom)/(1+imp_part*imp_charge)  )*knockon.get_src(i, j)/ba_ctx.zeta1_jp12[j];

		}
	}

};

/*
 * Set the nonstiff part  as explicit
 * @xx the unknown variables; @ff the rhs values to be returned
 */
template <class knockon_type>
void fk_equ_ba<knockon_type>::EvalNStiff(Field **xx, Field **ff, AppCtx *user)
{
	PetscInt i, j, k;
	PetscInt   nn = 0;
	PetscReal df;

	//	if ((! std::strcmp(user->ts_type, "explicit")) && my_mesh->with_trap)
	//		compute_Ftp(xx);
	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		if (j >= my_mesh->My2)
			xij = my_mesh->yf[j]; //0.5*(my_mesh->yy[j] + xic);
		else
			xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);

		xijp12 = my_mesh->yy[j];

		if (j < my_mesh->My2)
			xijp1 = my_mesh->yf[j];
		else
			xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j+1]) : (1.0);

		for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
			// velocities evaluated at j, at j+1 and j+1/2.
			vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
			vip12 = my_mesh->xx[i];
			vip1 =  my_mesh->xf[i];

			ff[j][i].fn[nn] = 0.0;

			if (! std::strcmp(user->ts_type, "explicit") ) {
                PetscPrintf(PETSC_COMM_WORLD,"======CHECK the integrator!!======\n");

				// energy flux
				ff[j][i].fn[nn] -= (Fv(nn, i, j, xx, PETSC_TRUE) - Fv(nn, i-1, j, xx, PETSC_TRUE))/vip12 /(vip1-vi);  //solving v*f

				// pitch-angle flux, it needs special attendtion at the trap-passing bounary
				if (j == my_mesh->My1) {
					// the non-local flux Ftp is computed with distribution from previous iteraction
					ff[j][i].fn[nn] -= (Fxi(nn, i, j, xx, PETSC_TRUE) - Fxi(nn, i, j-1, xx, PETSC_TRUE) + Ftp[i-my_mesh->xs+1]) /(xijp1-xij)/ba_ctx.zeta1_jp12[j];

				} else if (j == my_mesh->My2) {
					ff[j][i].fn[nn] -= (Fxi(nn, i, j, xx, PETSC_TRUE) - Ftp[i-my_mesh->xs+1]) /(xijp1-xij)/ba_ctx.zeta1_jp12[j];

				} else {
					ff[j][i].fn[nn] += (Fxi(nn, i, j, xx, PETSC_TRUE) - Fxi(nn, i, j-1, xx, PETSC_TRUE)) /(xijp1-xij)/ba_ctx.zeta1_jp12[j];
				}
				// add the knockon source term
				//				ff[j][i].fn[nn] += knockon.Eval(vip12, xijp12, i, j)/ba_ctx.zeta1_jp12[j]; //ba_ctx.zeta1(xijp12);
				ff[j][i].fn[nn] += knockon.get_src(i, j)/ba_ctx.zeta1_jp12[j];
			}
		}
	}
};


/* function to compute the xi flux on cell boundary*/
template <class knockon_type>
void fk_equ_ba<knockon_type>::Compute_Fxi(Field **xx)
{
	int nn = 0;
	PetscScalar coef, f(0.0), flux, E;
	PetscScalar xijp1, vip12;
	PetscScalar yr, yu, yd;
    pstate      state=E_Field->get_state();

	for (int j=my_mesh->ys-1; j<my_mesh->ys+my_mesh->ym; j++) { //xi, starting one cell below to cover both ends

        if (j == my_mesh->My1)
        {
            //the flux at j=My1 should never be used
			for(int i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) 
                fxi[(j-my_mesh->ys+1)*my_mesh->xm + i-my_mesh->xs]=NAN;

            continue;
        }
		if (j<0)
			xijp1 = -1.;
		else if (j < my_mesh->My2)
			xijp1 = my_mesh->yf[j];   // cell upper face
		else
			xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j+1]) : (1.0);

		if (xijp1>=-ba_ctx.xic && xijp1<=ba_ctx.xic)
			E = 0.;
		else
			E = state.E; 

        //xijp1=0 is another boundary
		if (xijp1>-1. && xijp1<1. && fabs(xijp1) > 1.e-16) { // avoid boundaries

			for(int i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
				vip12 = my_mesh->xx[i];

				coef = -(E/vip12) + state.alpha/sqrt(1.+vip12*vip12)*xijp1 * (ba_ctx.zeta2_jp1[j] + 4.*ba_ctx.zeta3_jp1[j]);

				if (coef < 0.) {
					if (j == my_mesh->My-2 || j == my_mesh->My2-2 || j == my_mesh->My1-1)
						f = (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
					else {
						yr = my_mesh->yy[j+2]; yu = my_mesh->yy[j+1]; yd = my_mesh->yy[j];
						f = Uface(xx[j+2][i].fn[nn], xx[j+1][i].fn[nn], xx[j][i].fn[nn], yr, yu, yd, xijp1, diff_type);
					}

				} else {
					if (j == my_mesh->My2 || j == my_mesh->My1+1 || j == 0) { //at the first +1 passing point, upwind
                        //at the first +1 passing point, upwind
                        // or right above trap-passing, already excluded being My2-1, so j+1 remains below xi=0
						f = (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
					} else {
						yr = my_mesh->yy[j-1]; yu = my_mesh->yy[j]; yd = my_mesh->yy[j+1];
						f = Uface(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], yr, yu, yd, xijp1, diff_type);
					}
				}

                flux=coef*f;

                coef=(ba_ctx.zeta2_jp1[j] -2.*ba_ctx.zeta3_jp1[j]) * dTv[i-my_mesh->xs];

                if (true)
                {
                    //a high-order diffusion. Here things are slightly more complicated than the slab case
                    if (j == 0) {
                        flux += -coef * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1, my_mesh->yy[j+2]-xijp1);
				    } else if (j<my_mesh->My2-2){
				    	flux += -coef * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        my_mesh->yy[j-1]-xijp1, my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1, my_mesh->yy[j+2]-xijp1);
                    } else if (j==my_mesh->My2-2){
				    	flux += -coef * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], 
                        my_mesh->yy[j-1]-xijp1, my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1);
                    }
                    else if (j==my_mesh->My2-1){
                        //symmetric bounary conditione is applied here
				    	flux += 0.0;
                    }
                    else if (j==my_mesh->My2){
                        flux += -coef * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn],
                        my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1, my_mesh->yy[j+2]-xijp1);
                    }
				    else if (j<my_mesh->My-2){
				    	flux += -coef * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        my_mesh->yy[j-1]-xijp1, my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1, my_mesh->yy[j+2]-xijp1);
                    } 
                    else{
				    	flux += -coef * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], 
                                my_mesh->yy[j-1]-xijp1, my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1);
                    }

                    if( false)// (j==my_mesh->My-2 || j==my_mesh->My-1 || j==my_mesh->My)
                    {
                        //note xx[My][i] is not defined
                        PetscPrintf(PETSC_COMM_WORLD,"j=%d, yy=%e, fn=%e\n", j, my_mesh->yy[j+2], xx[j+2][i].fn[nn]);
                        MPI_Barrier(PETSC_COMM_WORLD);
                        abort();
                    }
                }
                else {
                    //a low-order version
				    if (j == 0) {
                        flux += -coef * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j], my_mesh->yy[j+2]-my_mesh->yf[j]);
				    } else if (j<my_mesh->My2)
				    	flux += -coef * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], my_mesh->yy[j-1]-my_mesh->yf[j], my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j]);
				    else
				    	flux += -coef * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], my_mesh->yy[j-1]-my_mesh->yf[j+1], my_mesh->yy[j]-my_mesh->yf[j+1], my_mesh->yy[j+1]-my_mesh->yf[j+1]);
                }

				if (dw_ && true) {   // with quasilinear diffusion
					double D, fluxD, fluxe, fluxw;
					int idx1, idx2;

					if (j == my_mesh->My1-1) { // use the coefficient on the passing side (I do not understand this -QT)
						idx1= (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
						idx2 = idx1;
					} else {
						idx1= (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
						idx2 = (i-my_mesh->xs+1) + (j+1-my_mesh->ys+1)*(my_mesh->xm+2);
					}

					// d^2f/dxi^2
					D = 0.5*(Dw.xixi[idx1] + Dw.xixi[idx2]);

                    if (j == 0) {
			            fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1, my_mesh->yy[j+2]-xijp1);
                    } else if (j<my_mesh->My2-2) {
			            fluxD = -D/(vip12*vip12) * Uface.dfdx( xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                                my_mesh->yy[j-1]-xijp1, my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1, my_mesh->yy[j+2]-xijp1);
                    } else if (j==my_mesh->My2-2){
				    	fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], 
                        my_mesh->yy[j-1]-xijp1, my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1);
                    } else if (j==my_mesh->My2-1){
                        //symmetric bounary conditione is applied here
				    	fluxD = 0.0;
                    } else if (j==my_mesh->My2){
                        fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn],
                        my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1, my_mesh->yy[j+2]-xijp1);
                    } else if (j<my_mesh->My-2){
				    	fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        my_mesh->yy[j-1]-xijp1, my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1, my_mesh->yy[j+2]-xijp1);
                    } 
                    else{
				    	fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], 
                                my_mesh->yy[j-1]-xijp1, my_mesh->yy[j]-xijp1, my_mesh->yy[j+1]-xijp1);
                    }


					D = 0.5*(Dw.pxi[idx1] + Dw.pxi[idx2]);
                    
                    // d^2f/dxidp term (a central scheme)
                    // In numerical flux, we compute df/dp at xi=yf[j] and vi=xx[i]
			        if (i > 0 && i < (my_mesh->Mx-1)) {
                         double fim1, fi, fip1, fip2;

                         //linear interpolation 
                         if (j!=my_mesh->My2-1)
                         {
                         fim1=(xx[j+1][i-1].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i-1].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
                           fi=(xx[j+1][i  ].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i  ].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
                         fip1=(xx[j+1][i+1].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i+1].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
                         fip2=(xx[j+1][i+2].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i+2].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
                         }
                         else {
                         fim1=(xx[j][i-1].fn[nn] * (xijp1-my_mesh->yy[j-1]) + xx[j-1][i-1].fn[nn] * (my_mesh->yy[j]-xijp1)) / (my_mesh->yy[j]-my_mesh->yy[j-1]);
                           fi=(xx[j][i  ].fn[nn] * (xijp1-my_mesh->yy[j-1]) + xx[j-1][i  ].fn[nn] * (my_mesh->yy[j]-xijp1)) / (my_mesh->yy[j]-my_mesh->yy[j-1]);
                         fip1=(xx[j][i+1].fn[nn] * (xijp1-my_mesh->yy[j-1]) + xx[j-1][i+1].fn[nn] * (my_mesh->yy[j]-xijp1)) / (my_mesh->yy[j]-my_mesh->yy[j-1]);
                         fip2=(xx[j][i+2].fn[nn] * (xijp1-my_mesh->yy[j-1]) + xx[j-1][i+2].fn[nn] * (my_mesh->yy[j]-xijp1)) / (my_mesh->yy[j]-my_mesh->yy[j-1]);
 
                         }

		                 flux += fluxD - D/vip12*Uface.dfdx(fim1, fi, fip1, fip2, 
                         my_mesh->xx[i-1]-my_mesh->xx[i], 0.0, my_mesh->xx[i+1]-my_mesh->xx[i], my_mesh->xx[i+2]-my_mesh->xx[i]);
			        } else if (i == 0 || i >= (my_mesh->Mx-1)) {
			        	flux += fluxD - D/vip12 * 0.5*(xx[j+1][i+1].fn[nn] + xx[j][i+1].fn[nn] - xx[j+1][i].fn[nn] - xx[j][i].fn[nn]) 
                                                                / (my_mesh->xx[i+1] - my_mesh->xx[i]);;
			        }

                    //upwind for the extra term
		            if (D<0.) {
			            if (j == my_mesh->My-2)
			                flux += D/(vip12*vip12) * 
                                    (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) 
                                    / (my_mesh->yy[j+1]-my_mesh->yy[j]);
			            else 
                                    {
			                yr = my_mesh->yy[j+2]; yu = my_mesh->yy[j+1]; yd = my_mesh->yy[j];
			                flux += D/(vip12*vip12) *
                                    Uface(xx[j+2][i].fn[nn], xx[j+1][i].fn[nn], xx[j][i].fn[nn], yr, yu, yd, xijp1, diff_type);
			            }
		            } else {
		            	if (j == 0)
		            	    flux+= D/(vip12*vip12) *
                                   (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) 
                                   / (my_mesh->yy[j+1]-my_mesh->yy[j]);
		            	else {
		            	    yr = my_mesh->yy[j-1]; yu = my_mesh->yy[j]; yd = my_mesh->yy[j+1];
			                flux += D/(vip12*vip12) *
		            	            Uface(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], yr, yu, yd, xijp1, diff_type);
		            	}
		            }

				}
                else if (dw_ && false) {   // with quasilinear diffusion
					double dfdp, D, fluxe, fluxw;
					int idx1, idx2;

					if (j == my_mesh->My1-1) { // use the coefficient on the passing side
						idx1= (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
						idx2 = idx1;
					} else {
						idx1= (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
						idx2 = (i-my_mesh->xs+1) + (j+1-my_mesh->ys+1)*(my_mesh->xm+2);
					}

					// d^2f/dxi^2
					D = 0.5*(Dw.xixi[idx1] + Dw.xixi[idx2]);
					fluxe = -(D/(vip12*vip12)) * (xx[j+1][i].fn[nn] - xx[j][i].fn[nn])/( my_mesh->yy[j+1] - my_mesh->yy[j]);

					// the off-diagonal coefficient
					D = 0.5*(Dw.pxi[idx1] + Dw.pxi[idx2]);

					fluxw = fluxe;
					// d^2f/dxidp term, upwind
					if (i > 0) {
						if (D<0) {
							fluxe -= D/vip12 * (xx[j+1][i+1].fn[nn] - xx[j+1][i].fn[nn])/(my_mesh->xx[i+1] - my_mesh->xx[i]);
							fluxw -= D/vip12 * (xx[j][i].fn[nn] - xx[j][i-1].fn[nn])/(my_mesh->xx[i] - my_mesh->xx[i-1]);
						} else {
							fluxe -= D/vip12 * (xx[j+1][i].fn[nn] - xx[j+1][i-1].fn[nn])/(my_mesh->xx[i] - my_mesh->xx[i-1]);
							fluxw -= D/vip12 * (xx[j][i+1].fn[nn] - xx[j][i].fn[nn])/(my_mesh->xx[i+1] - my_mesh->xx[i]);
						}

						if (fluxe*fluxw > 0)
							flux += 2.*fluxw* fabs(fluxe)/(fabs(fluxe)+fabs(fluxw));

					} else if (i == 0) {
						fluxe -= D/vip12 * 0.5*(xx[j+1][i+1].fn[nn] + xx[j][i+1].fn[nn] - xx[j+1][i].fn[nn] - xx[j][i].fn[nn]) / (my_mesh->xx[i+1] - my_mesh->xx[i]);
						flux += fluxe;
					}

					// convective flux from quasilinear term due to using f*p
					if (D>0)
						flux += D/(vip12*vip12) * 0.5*(xx[j][i].fn[nn] + xx[j][i].fn[nn]);
					else
						flux += D/(vip12*vip12) * 0.5*(xx[j+1][i].fn[nn] + xx[j+1][i].fn[nn]);
				}

				fxi[(j-my_mesh->ys+1)*my_mesh->xm + i-my_mesh->xs] = (1 - xijp1*xijp1) * flux; //(j-my_mesh->ys+1)*my_mesh->xm + i-my_mesh->xs
			}//end i
		}
	}//end j

};

/* function to compute the energy flux on cell boundary*/
template <class knockon_type>
void fk_equ_ba<knockon_type>::Compute_Fv(Field **xx)
{
	int nn = 0;
	PetscScalar f(0.0), flux, coef, E;
	PetscScalar xijp12, vip1;
	PetscScalar xr, xu, xd;
    pstate      state=E_Field->get_state();

	for (int j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi, starting one cell below to cover both ends

		xijp12 = my_mesh->yy[j];  // cell center

		if (xijp12>=-ba_ctx.xic && xijp12<=ba_ctx.xic)
			E = 0.;
		else
			E = state.E;

		for(int i=my_mesh->xs-1; i<my_mesh->xs+my_mesh->xm; i++) { //v
			if (i == -1 )
				vip1 = my_mesh->xmin;
			else
				vip1 =  my_mesh->xf[i];

			PetscScalar aTv_ = aTv[i-my_mesh->xs+1]; 
			PetscScalar bTv_ = bTv[i-my_mesh->xs+1]; 

			coef = -vip1*( E*xijp12/ba_ctx.zeta1_jp12[j] + state.alpha*vip1*sqrt(1.+vip1*vip1)*( (1. - xijp12*xijp12) + quad(rho)*vip1*vip1*quad(xijp12)) * (1. + 6.*ba_ctx.zeta4_jp12[j]/ba_ctx.zeta1_jp12[j]) );
			coef -= (bTv_*vip1 - aTv_); //remember to change here if you want to solve f, pf, p^2f

			/* energy convective flux */
			if (bc_type == 1) {  // zero flux high-energy boundary
				if (coef < 0.) {
					if (i == my_mesh->Mx - 1) {
						f = 0.0;
					} else if (i == my_mesh->Mx - 2) {
						xr = my_mesh->xf[i+1]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
						f = Uface(0., xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type);
					} else if (i == -1) {
						xr = my_mesh->xx[1]; xu = my_mesh->xx[0]; xd = 2.*my_mesh->xmin - xu;
						f = Uface(xx[j][1].fn[nn], xx[j][0].fn[nn], (ra*Fmax(xd)*xd), xr, xu, xd, vip1, diff_type);
					} else {
						xr = my_mesh->xx[i+2]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
						f = Uface(xx[j][i+2].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type);
					}
				} else {
					if (i == my_mesh->Mx - 1) {
						f = 0.0; 
					} else if (i == 0) {
						xu = my_mesh->xx[0]; xd = my_mesh->xx[1]; xr = 2.*my_mesh->xmin - xu;
						f = Uface(((ra*Fmax(xr))*xr), xx[j][0].fn[nn], xx[j][1].fn[nn], xr, xu, xd, vip1, diff_type);
					} else if (i == -1) {
						xd = my_mesh->xx[0]; xr = (4.*my_mesh->xmin - 3.*xd); xu = (2.*my_mesh->xmin - xd);
						f = Uface((ra*Fmax(xr)*xr), (ra*Fmax(xu)*xu), xx[j][0].fn[nn], xr, xu, xd, vip1, diff_type);
					} else {
						xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xx[i+1];
						f = Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xr, xu, xd, vip1, diff_type);
					}
				}
				flux = coef*f;

			} else {  // finite outgoing flux
                PetscPrintf(PETSC_COMM_WORLD,"======CHECK ME!======\n");
				if (coef < 0.) {
					if (i == my_mesh->Mx - 1) {
						f = 0.; //Uface(0.0, 0.0, xx[j][i].fn[nn]);//(3.0/8.0)*xx[j][i].fn[nn];
					} else if (i == my_mesh->Mx - 2) {
						xr = my_mesh->xf[i+1]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
						f = Uface(xx[j][i+1].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type); //Uface(0.0, xx[j][i+1].fn[nn], xx[j][i].fn[nn]); //Uface(0.0, xx[j][i+1].fn[nn], xx[j][i].fn[nn]); //(3.0/8.0)*xx[j][i].fn[nn] + (6.0/8.0)*xx[j][i+1].fn[nn];//0.5*(xx[j][i].fn[nn] + xx[j][i+1].fn[nn]);
					} else if (i == -1) {
						xr = my_mesh->xx[1]; xu = my_mesh->xx[0]; xd = 2.*my_mesh->xmin - xu;
						f = Uface(xx[j][1].fn[nn], xx[j][0].fn[nn], (ra*Fmax(xd)*xd), xr, xu, xd, vip1, diff_type);
					} else {
						xr = my_mesh->xx[i+2]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
						f = Uface(xx[j][i+2].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type);
					}
				} else {
					if (i == (my_mesh->Mx - 1)) {
						//				xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xmax;
						f = xx[j][i].fn[nn]; //Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], 0.0, xr, xu, xd, vip1); //Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], 0.0); //(6.0/8.0)*xx[j][i].fn[nn] + (-1.0/8.0)*xx[j][i-1].fn[nn];
					} else if (i == 0) {
						xu = my_mesh->xx[0]; xd = my_mesh->xx[1]; xr = 2.*my_mesh->xmin - xu;
						f = Uface(((ra*Fmax(xr))*xr), xx[j][0].fn[nn], xx[j][1].fn[nn], xr, xu, xd, vip1, diff_type);
					} else if (i == -1) {
						xd = my_mesh->xx[0]; xr = (4.*my_mesh->xmin - 3.*xd); xu = (2.*my_mesh->xmin - xd);
						f = Uface((ra*Fmax(xr)*xr), (ra*Fmax(xu)*xu*xu), xx[j][0].fn[nn], xr, xu, xd, vip1, diff_type);
					} else {
						xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xx[i+1];
						f = Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xr, xu, xd, vip1, diff_type);
					}
				}
				flux = coef*f;
			}

			// energy diffusive flux due to collision (central differencing)
			if (i == -1) {
				xd = (2.*my_mesh->xmin - my_mesh->xx[0]);
				flux -= vip1*aTv_ * (xx[j][0].fn[nn] - (ra*Fmax(xd)*xd))/(my_mesh->xx[0] - xd) ;
			} else if (i == 0) {
				flux -= vip1*aTv_ * Uface.dfdx(xx[j][i].fn[nn], xx[j][i+1].fn[nn], xx[j][i+2].fn[nn], my_mesh->xx[i]-my_mesh->xf[i], my_mesh->xx[i+1]-my_mesh->xf[i], my_mesh->xx[i+2]-my_mesh->xf[i]);
			} else if (i < (my_mesh->Mx-1)) {
                if (true)
                {
                    //high-order diffusion 
                    double xface=my_mesh->xf[i];
		            flux -= vip1*aTv_ * Uface.dfdx(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xx[j][i+2].fn[nn],
                       my_mesh->xx[i-1]-xface, my_mesh->xx[i]-xface, my_mesh->xx[i+1]-xface, my_mesh->xx[i+2]-xface);
                }
                else
				    flux -= vip1*aTv_ * Uface.dfdx(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], my_mesh->xx[i-1]-my_mesh->xf[i], my_mesh->xx[i]-my_mesh->xf[i], my_mesh->xx[i+1]-my_mesh->xf[i]);

				// with quasilinear diffusion
                if (dw_ && true) {
					double Dpp, Dpxi, fluxD;
                    double xface=my_mesh->xf[i];

					int idx1 = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
					int idx2 = (i+1-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);

					// the diagonal coefficient
					Dpp = 0.5*(Dw.pp[idx1] + Dw.pp[idx2]) * (1.-xijp12*xijp12)/ba_ctx.zeta1_jp12[j];
					// the off-diagonal coefficient
					Dpxi = 0.5*(Dw.pxi[idx1] + Dw.pxi[idx2])*(1.-xijp12*xijp12)/ba_ctx.zeta1_jp12[j];

					// d^2/dp^2 term
			        fluxD = -vip1*Dpp * Uface.dfdx(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xx[j][i+2].fn[nn],
                            my_mesh->xx[i-1]-xface, my_mesh->xx[i]-xface, my_mesh->xx[i+1]-xface, my_mesh->xx[i+2]-xface);

					// d^2/dpdxi term, upwind
                    if (j>0 && j<my_mesh->My-2) {
                        double fim1, fi, fip1, fip2;

                        //linear interpolation
                        fim1=(xx[j-1][i+1].fn[nn] * (xface-my_mesh->xx[i]) + xx[j-1][i].fn[nn] * (my_mesh->xx[i+1]-xface)) / (my_mesh->xx[i+1]-my_mesh->xx[i]);
                          fi=(xx[j  ][i+1].fn[nn] * (xface-my_mesh->xx[i]) + xx[j  ][i].fn[nn] * (my_mesh->xx[i+1]-xface)) / (my_mesh->xx[i+1]-my_mesh->xx[i]);
                        fip1=(xx[j+1][i+1].fn[nn] * (xface-my_mesh->xx[i]) + xx[j+1][i].fn[nn] * (my_mesh->xx[i+1]-xface)) / (my_mesh->xx[i+1]-my_mesh->xx[i]);
                        fip2=(xx[j+2][i+1].fn[nn] * (xface-my_mesh->xx[i]) + xx[j+2][i].fn[nn] * (my_mesh->xx[i+1]-xface)) / (my_mesh->xx[i+1]-my_mesh->xx[i]);

		                 flux += fluxD - Dpxi*Uface.dfdx(fim1, fi, fip1, fip2, 
                                         my_mesh->yy[j-1]-my_mesh->yy[j], 0.0, my_mesh->yy[j+1]-my_mesh->yy[j], my_mesh->yy[j+2]-my_mesh->yy[j]);
                    } else if (j==0 || j==my_mesh->My-2) { 
                        //low-order approximation
				        flux += fluxD - Dpxi * 0.5*(xx[j+1][i+1].fn[nn] + xx[j+1][i].fn[nn] - xx[j][i+1].fn[nn] - xx[j][i].fn[nn]) 
                                                     / (my_mesh->yy[j+1] - my_mesh->yy[j]);
			        }

					// quasilinear convective flux due to using f*p, upwinded (note the negative sign)
		            if (Dpp < 0.) {
			            if (i >= my_mesh->Mx - 2 || i==-1) {
                            //the boundary condition is not implemented for now
				            flux += Dpp * xx[j][i+1].fn[nn];
			            } else {
				            xr = my_mesh->xx[i+2]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
				            flux += Dpp * Uface(xx[j][i+2].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type);
			            }
		            } else {
		            	if (i == my_mesh->Mx - 1 || i <= 0) {
				            flux += Dpp * xx[j][i].fn[nn];
		            	} else {
		            		xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xx[i+1];
		            		flux += Dpp * Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xr, xu, xd, vip1, diff_type);
		            	}
		            }

				}
                else if (dw_ && false) {
					double Dpp, Dpxi, fluxe, fluxw;
					int idx1 = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
					int idx2 = (i+1-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);

					Dpp = 0.5*(Dw.pp[idx1] + Dw.pp[idx2]) * (1.-xijp12*xijp12);

					// d^2/dp^2 term
					fluxe = -vip1*Dpp * (xx[j][i+1].fn[nn] - xx[j][i].fn[nn])/(my_mesh->xx[i+1] - my_mesh->xx[i]);

					fluxw = fluxe;

					// the off-diagonal coefficient
					Dpxi = 0.5*(Dw.pxi[idx1] + Dw.pxi[idx2])*(1.-xijp12*xijp12);

					// d^2/dpdxi term, upwind
					if (j>0 && j<my_mesh->My-1) {
						if (Dpxi<0) {
							fluxe -= Dpxi * (xx[j+1][i+1].fn[nn] - xx[j][i+1].fn[nn])/(my_mesh->yy[j+1] - my_mesh->yy[j]);
							fluxw -= Dpxi * (xx[j][i].fn[nn] - xx[j-1][i].fn[nn])/(my_mesh->yy[j] - my_mesh->yy[j-1]);
						} else {
							fluxe -= Dpxi * (xx[j][i+1].fn[nn] - xx[j-1][i+1].fn[nn])/(my_mesh->yy[j] - my_mesh->yy[j-1]);
							fluxw -= Dpxi * (xx[j+1][i].fn[nn] - xx[j][i].fn[nn])/(my_mesh->yy[j+1] - my_mesh->yy[j]);
						}

						if (fluxe*fluxw > 0)
							flux += 2.*fluxw*fabs(fluxe)/(fabs(fluxe)+fabs(fluxw));

					} else if (j == 0) { // one side derivative
						fluxe -= Dpxi * 0.5*(xx[j+1][i+1].fn[nn] + xx[j+1][i].fn[nn] - xx[j][i+1].fn[nn] - xx[j][i].fn[nn]) / (my_mesh->yy[j+1] - my_mesh->yy[j]);
						flux += fluxe;
					}

					// quasilinear convective flux due to using f*p, upwinded (note the negative sign)
					if (Dpp<0)
						flux += Dpp * xx[j][i+1].fn[nn];
					else
						flux += Dpp * xx[j][i].fn[nn];
				}
			}

			fp[(j-my_mesh->ys)*(my_mesh->xm+1) + i-my_mesh->xs+1] = flux;

		}
	}
};


// flux leaking to the counter passing region
template <class knockon_type>
void fk_equ_ba<knockon_type>::compute_Ftp(Field **xx)
{
	PetscScalar coef, f(0.0), flux;
	PetscScalar vip12;



    //the interface is located at yf[my_mesh->My1] which is interpolated from the inner two points in the trapped region
	double d1 = (my_mesh->yf[my_mesh->My1] - my_mesh->yy[my_mesh->My1]), d2 = (my_mesh->yy[my_mesh->My1] - my_mesh->yy[my_mesh->My1-1]);

	for(int i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) {
		f = xx[my_mesh->My1][i].fn[0] * (1.+d1/d2) - xx[my_mesh->My1-1][i].fn[0] * d1/d2 ; // approximate the function at the + trap-passing boundary

        //compute the flux at y=yf[my_mesh->My2] x=xx[i]
        //see Dwp_ba::setupFace for details
        double xijp1=my_mesh->yf[my_mesh->My2];
		vip12 = my_mesh->xx[i];

		// trap->passing +xitp at j=My2
		coef = sr/sqrt(1.+vip12*vip12)*xijp1 * ba_ctx.zeta2_jp1[my_mesh->My2];
		flux = ( coef * f - ba_ctx.zeta2_jp1[my_mesh->My2] * dTv[i-my_mesh->xs] * (xx[my_mesh->My2][i].fn[0] - f)/(my_mesh->yy[my_mesh->My2] - xijp1) );

		if (dw_) {   // with quasilinear diffusion
			double D, fluxD;
			int idx = (i-my_mesh->xs+1) + 0*(my_mesh->xm+1); // index of the diffusion coefficient matrix

			// d^2f/dxi^2
			D = Dw.xixiFace[idx];
			fluxD = -(D/(vip12*vip12)) * (xx[my_mesh->My2][i].fn[0] - f) / (my_mesh->yy[my_mesh->My2]-xijp1);

			// the off-diagonal coefficient
			D = Dw.pxiFace[idx];

			// d^2f/dxidp term
			if (i > 0) {
				double f1 = xx[my_mesh->My1][i-1].fn[0] * (1.+d1/d2) - xx[my_mesh->My1-1][i-1].fn[0] * d1/d2 ;
				fluxD -= D/vip12 * (f - f1)/(my_mesh->xx[i] - my_mesh->xx[i-1]);
				flux += fluxD;

			} else if (i == 0) {
				double f1 = xx[my_mesh->My1][i+1].fn[0] * (1.+d1/d2) - xx[my_mesh->My1-1][i+1].fn[0] * d1/d2 ;
				fluxD -= D/vip12 * (f1 - f)/(my_mesh->xx[i+1] - my_mesh->xx[i]);
				flux += fluxD;
			}

			// convective flux from quasilinear term due to using f*p
			flux += D/(vip12*vip12) * f;
		}
		Ftp2[i-my_mesh->xs+1] = flux * (1 - xijp1*xijp1);

        //compute flux similarly at xi=yf[my_mesh->My1] 
        xijp1=my_mesh->yf[my_mesh->My1];

		//passing->trap -xitp, notice the factor of 2 to account for change of Jacobian
		coef = sr/sqrt(1.+vip12*vip12)*xijp1 * ba_ctx.zeta2_jp1[my_mesh->My1];
		flux = ( coef * f  - ba_ctx.zeta2_jp1[my_mesh->My1] * dTv[i-my_mesh->xs] * (xx[my_mesh->My1+1][i].fn[0] - f)/( my_mesh->yy[my_mesh->My1+1] - xijp1 ) );

		if (dw_) {   // with quasilinear diffusion
			double dfdp, D, fluxD;

			int idx = (i-my_mesh->xs+1) + 1*(my_mesh->xm+1); // index of the diffusion coefficient matrix

			// d^2f/dxi^2
			D = Dw.xixiFace[idx];
			fluxD = -(D/(vip12*vip12)) * (xx[my_mesh->My1+1][i].fn[0] - f)/(my_mesh->yy[my_mesh->My1+1]-xijp1);

			// the off-diagonal coefficient
			D = Dw.pxiFace[idx];

			// d^2f/dxidp term
			if (i > 0) {
				double f1 = xx[my_mesh->My1][i-1].fn[0] * (1.+d1/d2) - xx[my_mesh->My1-1][i-1].fn[0] * d1/d2 ;
				fluxD -= D/vip12 * (f - f1)/(my_mesh->xx[i] - my_mesh->xx[i-1]);
				flux += fluxD;
			} else if (i == 0) {
				double f1 = xx[my_mesh->My1][i+1].fn[0] * (1.+d1/d2) - xx[my_mesh->My1-1][i+1].fn[0] * d1/d2 ;
				fluxD -= D/vip12 * (f1 - f)/(my_mesh->xx[i+1] - my_mesh->xx[i]);
				flux += fluxD;
			}

			// convective flux from quasilinear term due to using f*p
			flux += D/(vip12*vip12) * f;
		}
		Ftp1[i-my_mesh->xs+1] = flux * (1 - xijp1*xijp1);
	}

};


/* Set pre-conditioner for the implicit part (dF/dX + a dF/dXdot) of the IMEX scheme*/
template <class knockon_type>
void fk_equ_ba<knockon_type>::SetIJacobian(Field **x, Field **xdot, PetscReal a, Mat jac, AppCtx *user)
{
	PetscErrorCode ierr;
	int         	i,j;

	MatStencil       col[3],row;
	PetscScalar      v[3];

	PetscReal        aTv_, bTv_, cTv_, dTv_, aTxi, cTxi, E, temp;
	PetscReal        av,cv;
	PetscReal        axi,cxi;
    pstate           state=E_Field->get_state();

	E = state.E;

	PetscReal hx = 0.1, hy = 0.1;

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) {

		xij = (PetscScalar)(-1.0 + j*hy);
		xijp12 = (PetscScalar)(-1.0 + (j+0.5)*hy);
		xijp1 = (PetscScalar)(-1.0 + (j+1)*hy);

		for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) {

			row.j = j; row.i = i;

			// velocities evaluated at j, at j+1 and j+1/2.
			vi = (PetscScalar)(my_mesh->xmin + i*hx);
			vip12 = (PetscScalar)(my_mesh->xmin + (i+0.5)*hx);
			vip1 = (PetscScalar)(my_mesh->xmin + (i+1.0)*hx);

			// pitch-angle diffusion
			dTv_ = -dTv[i-my_mesh->xs] *hx/hy;
			aTxi = (1-xijp1*xijp1)*dTv_;
			cTxi = (1-xij*xij)*dTv_;

			// energy diffusion
			aTv_ = -sqrt(1+vip1*vip1)/vip12*Chand(vip1) *hy/hx;
			cTv_ = -sqrt(1+vi*vi)/vip12*Chand(vi) *hy/hx;

			// pitch angle flux due to electric field, radiation
			axi = -(E/vip12) + sr/sqrt(1+vip12*vip12)*xijp1;
			axi *= (1 - xijp1*xijp1) *hx;

			cxi = -(E/vip12) + sr/sqrt(1+vip12*vip12)*xij;
			cxi *= (1 - xij*xij) *hx;

			/* convective energy flux */
			av = -( E*xijp12 + sr*vip1*sqrt(1+vip1*vip1)*( (1 - xijp12*xijp12) + quad(rho)*vip1*vip1*quad(xijp12) ) ); // electric field + radiation caused flux
			av -= (1.0/Dpp - sqrt(1+vip1*vip1)/vip1/vip1)  * Chand(vip1); // collisional drag
			av *= vip1/vip12 *hy;

			cv = -( E*xijp12 + sr*vi*sqrt(1+vi*vi)*( (1 - xijp12*xijp12) + quad(rho)*vi*vi*quad(xijp12) ) ); // electric field + radiation caused flux
			cv -= (1.0/Dpp - sqrt(1+vi*vi)/vi/vi) * Chand(vi); // collisional drag
			cv *= vi/vip12 *hy;

			// set Jacobian elements
			if (i == 0 ) {
				v[0] = 1.0;	                 col[0].j = j;   col[0].i = i;
				ierr = MatSetValuesStencil(jac,1,&row,1,col,v, ADD_VALUES);

			} else {
				/* -------------------------------------------------- energy fluxes ----------------------------------------------------*/

				if (i == 1) {
					if (av < 0) {
						v[0] = -aTv_ + (3.0/8.0)*av;    	    col[0].j = j;   col[0].i = i;
						v[1] = aTv_ + (3.0/4.0)*av;			    col[1].j = j;   col[1].i = i+1;
						v[2] = -(1.0/8.0)*av;  	   			    col[2].j = j;   col[2].i = i+2;

						ierr = MatSetValuesStencil(jac,1,&row, 3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = -(1.0/8.0)*av;      			    col[0].j = j;   col[0].i = i-1;
						v[1] = -aTv_ + (3.0/4.0)*av;		    col[1].j = j;   col[1].i = i;
						v[2] = aTv_ + (3.0/8.0)*av;  	   	    col[2].j = j;   col[2].i = i+1;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
					if (cv < 0) {
						v[0] = cTv_ - (3.0/8.0)*cv;    		    col[0].j = j;   col[0].i = i-1;
						v[1] = -cTv_ - (3.0/4.0)*cv;		    col[1].j = j;   col[1].i = i;
						v[2] = (1.0/8.0)*cv;  	   			    col[2].j = j;   col[2].i = i+1;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = cTv_ - (3.0/4.0)*cv;			    col[0].j = j;   col[0].i = i-1;
						v[1] = -cTv_ - (3.0/8.0)*cv;  	   	    col[1].j = j;   col[1].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,2,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
				} else if (i == my_mesh->Mx-1) {
					if (cv < 0) {
						v[0] = cTv_ - (3.0/8.0)*cv;    					    col[0].j = j;   col[0].i = i-1;
						v[1] = -cTv_ - (3.0/4.0)*cv + (1.0/8.0)*cv;		    col[1].j = j;   col[1].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,2,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = (1.0/8.0)*cv;      			    col[0].j = j;   col[0].i = i-2;
						v[1] = cTv_ - (3.0/4.0)*cv;			    col[1].j = j;   col[1].i = i-1;
						v[2] = -cTv_ - (3.0/8.0)*cv;  	   	    col[2].j = j;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}

				} else if (i == my_mesh->Mx-2) {
					if (av < 0) {
						v[0] = -aTv_ + (3.0/8.0)*av;    				    col[0].j = j;   col[0].i = i;
						v[1] = aTv_ + (3.0/4.0)*av - (1.0/8.0)*av;		    col[1].j = j;   col[1].i = i+1;

						ierr = MatSetValuesStencil(jac,1,&row, 2,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = -(1.0/8.0)*av;      			    col[0].j = j;   col[0].i = i-1;
						v[1] = -aTv_ + (3.0/4.0)*av;		    col[1].j = j;   col[1].i = i;
						v[2] = aTv_ + (3.0/8.0)*av;  	   	    col[2].j = j;   col[2].i = i+1;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
					if (cv < 0) {
						v[0] = cTv_ - (3.0/8.0)*cv;    		    col[0].j = j;   col[0].i = i-1;
						v[1] = -cTv_ - (3.0/4.0)*cv;		    col[1].j = j;   col[1].i = i;
						v[2] = (1.0/8.0)*cv;  	   			    col[2].j = j;   col[2].i = i+1;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = (1.0/8.0)*cv;      			    col[0].j = j;   col[0].i = i-2;
						v[1] = cTv_ - (3.0/4.0)*cv;			    col[1].j = j;   col[1].i = i-1;
						v[2] = -cTv_ - (3.0/8.0)*cv;  	   	    col[2].j = j;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
				} else {
					if (av < 0) {
						v[0] = -aTv_ + (3.0/8.0)*av;    	    col[0].j = j;   col[0].i = i;
						v[1] = aTv_ + (3.0/4.0)*av;			    col[1].j = j;   col[1].i = i+1;
						v[2] = -(1.0/8.0)*av;  	   			    col[2].j = j;   col[2].i = i+2;

						ierr = MatSetValuesStencil(jac,1,&row, 3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = -(1.0/8.0)*av;      			    col[0].j = j;   col[0].i = i-1;
						v[1] = -aTv_ + (3.0/4.0)*av;		    col[1].j = j;   col[1].i = i;
						v[2] = aTv_ + (3.0/8.0)*av;  	   	    col[2].j = j;   col[2].i = i+1;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
					if (cv < 0) {
						v[0] = cTv_ - (3.0/8.0)*cv;    		    col[0].j = j;   col[0].i = i-1;
						v[1] = -cTv_ - (3.0/4.0)*cv;		    col[1].j = j;   col[1].i = i;
						v[2] = (1.0/8.0)*cv;  	   			    col[2].j = j;   col[2].i = i+1;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = (1.0/8.0)*cv;      			    col[0].j = j;   col[0].i = i-2;
						v[1] = cTv_ - (3.0/4.0)*cv;			    col[1].j = j;   col[1].i = i-1;
						v[2] = -cTv_ - (3.0/8.0)*cv;  	   	    col[2].j = j;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
				}

				/* -------------------------------------------------- pitch-angle fluxes ----------------------------------------------------*/

				if (j == 0) {
					if (axi < 0) {
						v[0] = -aTxi + (3.0/8.0)*axi;    	    col[0].j = j;     col[0].i = i;
						v[1] = aTxi + (3.0/4.0)*axi;		    col[1].j = j+1;   col[1].i = i;
						v[2] = -(1.0/8.0)*axi;  	   		    col[2].j = j+2;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row, 3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = -aTxi + 0.5*axi;		    col[0].j = j;    col[0].i = i;
						v[1] = aTxi + 0.5*axi;  	    col[1].j = j+1;  col[1].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,2,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
				} else if (j == 1) {
					if (axi < 0) {
						v[0] = -aTxi + (3.0/8.0)*axi;    	    col[0].j = j;     col[0].i = i;
						v[1] = aTxi + (3.0/4.0)*axi;		    col[1].j = j+1;   col[1].i = i;
						v[2] = -(1.0/8.0)*axi;  	   		    col[2].j = j+2;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row, 3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = -(1.0/8.0)*axi;      		    col[0].j = j-1;   col[0].i = i;
						v[1] = -aTxi + (3.0/4.0)*axi;		    col[1].j = j;     col[1].i = i;
						v[2] = aTxi + (3.0/8.0)*axi;  	   	    col[2].j = j+1;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
					if (cxi < 0) {
						v[0] = cTxi - (3.0/8.0)*cxi;    	    col[0].j = j-1;   col[0].i = i;
						v[1] = -cTxi - (3.0/4.0)*cxi;		    col[1].j = j;     col[1].i = i;
						v[2] = (1.0/8.0)*cxi;  	   			    col[2].j = j+1;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = cTxi - 0.5*cxi;	    	    col[0].j = j-1;  col[0].i = i;
						v[1] = -cTxi - 0.5*cxi;  	   	    col[1].j = j;    col[1].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,2, col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
				} else if (j == my_mesh->My-1) {
					if (cxi < 0) {
						v[0] = cTxi - 0.5*cxi;    	    col[0].j = j-1;  col[0].i = i;
						v[1] = -cTxi - 0.5*cxi;		    col[1].j = j;    col[1].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,2,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = (1.0/8.0)*cxi;      			    col[0].j = j-2;   col[0].i = i;
						v[1] = cTxi - (3.0/4.0)*cxi;		    col[1].j = j-1;   col[1].i = i;
						v[2] = -cTxi - (3.0/8.0)*cxi;  	   	    col[2].j = j;     col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}

				} else if (j == my_mesh->My-2) {
					if (axi < 0) {
						v[0] = -aTxi + 0.5*axi;		    col[0].j = j;    col[0].i = i;
						v[1] = aTxi + 0.5*axi;		    col[1].j = j+1;  col[1].i = i;

						ierr = MatSetValuesStencil(jac,1,&row, 2,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = -(1.0/8.0)*axi;      		    col[0].j = j-1;   col[0].i = i;
						v[1] = -aTxi + (3.0/4.0)*axi;		    col[1].j = j;     col[1].i = i;
						v[2] = aTxi + (3.0/8.0)*axi;  	   	    col[2].j = j+1;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
					if (cxi < 0) {
						v[0] = cTxi - (3.0/8.0)*cxi;    	    col[0].j = j-1;   col[0].i = i;
						v[1] = -cTxi - (3.0/4.0)*cxi;		    col[1].j = j;     col[1].i = i;
						v[2] = (1.0/8.0)*cxi;  	   			    col[2].j = j+1;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = (1.0/8.0)*cxi;      			    col[0].j = j-2;   col[0].i = i;
						v[1] = cTxi - (3.0/4.0)*cxi;		    col[1].j = j-1;   col[1].i = i;
						v[2] = -cTxi - (3.0/8.0)*cxi;  	   	    col[2].j = j;     col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
				} else {
					if (axi < 0) {
						v[0] = -aTxi + (3.0/8.0)*axi;    	    col[0].j = j;     col[0].i = i;
						v[1] = aTxi + (3.0/4.0)*axi;		    col[1].j = j+1;   col[1].i = i;
						v[2] = -(1.0/8.0)*axi;  	   		    col[2].j = j+2;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row, 3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = -(1.0/8.0)*axi;      		    col[0].j = j-1;   col[0].i = i;
						v[1] = -aTxi + (3.0/4.0)*axi;		    col[1].j = j;     col[1].i = i;
						v[2] = aTxi + (3.0/8.0)*axi;  	   	    col[2].j = j+1;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
					if (cxi < 0) {
						v[0] = cTxi - (3.0/8.0)*cxi;    	    col[0].j = j-1;   col[0].i = i;
						v[1] = -cTxi - (3.0/4.0)*cxi;		    col[1].j = j;     col[1].i = i;
						v[2] = (1.0/8.0)*cxi;  	   			    col[2].j = j+1;   col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					} else {
						v[0] = (1.0/8.0)*cxi;      			    col[0].j = j-2;   col[0].i = i;
						v[1] = cTxi - (3.0/4.0)*cxi;		    col[1].j = j-1;   col[1].i = i;
						v[2] = -cTxi - (3.0/8.0)*cxi;  	   	    col[2].j = j;     col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v, ADD_VALUES);//CHKERRQ(ierr);
					}
				}

			}
		}
	}
};


template class fk_equ_ba<knockon_rp_ba>;
template class fk_equ_ba<knockon_chiu_ba>;
template class fk_equ_ba<knockon_none_ba>;
