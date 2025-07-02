
/*
 * fk_equ_slab.cpp
 *
 *  Created on: Sep 25, 2014
 *      Author: zehuag
 */

#include "fk_equ_slab.h"
#include "knockon_rp.h"
#include "knockon_chiu.h"
#include "knockon_none.h"

/* constructor
 * reading parameters and set up the collision operator
 * */
template <class knockon_type>
fk_equ_slab<knockon_type>::fk_equ_slab(mesh *mesh_, Field_EQU *E_Field_, char* param_file) :
		fk_equ(mesh_, E_Field_, param_file),
		knockon(mesh_, fk_equ::beta, 10.0*fk_equ::vte), Dw(mesh_)
{
	int i, j, nn;
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

	PetscPrintf(PETSC_COMM_WORLD, "Running 2D RFK equation in slab geometry.\n");
};



/* An iterative method usually requires an initial guess for the function to be solved*/
template <class knockon_type>
void fk_equ_slab<knockon_type>::initialize(Field **xx, AppCtx *user)
{
	int i, j, nn;
	PetscReal v,xi;

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi
		xi = my_mesh->yy[j];

		for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { // v
			v = my_mesh->xx[i];

            if (user->flag_df){
			    xx[j][i].fn[0] = ra * Fmax(v)*v + ra * Delta(v,xi)*v;
            }
            else{
			    xx[j][i].fn[0] = ra * Fmax(v)*v;
            }
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
 * 
 */
template <class knockon_type>
void fk_equ_slab<knockon_type>::EvalStiff(Field **xx, Field **xdot, Field **ff, AppCtx *user)
{
	PetscInt i, j, k;
	PetscInt   nn = 0;
    PetscBool  FullImplicit=(PetscBool)(!strcmp(user->ts_type, "implicit"));

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);
		xijp12 = my_mesh->yy[j];
		xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j]) : (1.0);

		for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
			// velocities evaluated at j, at j+1 and j+1/2.
			vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
			vip12 = my_mesh->xx[i];
			vip1 = my_mesh->xf[i];

			// time derivative
			ff[j][i].fn[nn] = xdot[j][i].fn[nn];

			// energy flux
			ff[j][i].fn[nn] += (Fv(nn, i, j, xx, FullImplicit) - Fv(nn, i-1, j, xx, FullImplicit))/vip12 /(vip1-vi);  //solving v*f

			// pitch angle flux
			ff[j][i].fn[nn] += (Fxi(nn, i, j, xx, FullImplicit) - Fxi(nn, i, j-1, xx, FullImplicit)) /(xijp1-xij);

			// add the knockon source term
            // Emil removed the source term - remember to put it back!
			ff[j][i].fn[nn] -= ( (1+imp_part*imp_atom)/(1+imp_part*imp_charge)  )*knockon.get_src(i, j);
            
		}
	}

};


/*
 * Set the nonstiff part  as explicit
 * @xx the unknown variables; @ff the rhs values to be returned
 */
template <class knockon_type>
void fk_equ_slab<knockon_type>::EvalNStiff(Field **xx, Field **ff, AppCtx *user)
{
	PetscInt i, j, k;
	PetscInt   nn = 0;
    //printf("ts_type is %s\n", user->ts_type);

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);
		xijp12 = my_mesh->yy[j];
		xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j]) : (1.0);

		for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
			// velocities evaluated at j, at j+1 and j+1/2.
			vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
			vip12 = my_mesh->xx[i];
			vip1 = my_mesh->xf[i];

			ff[j][i].fn[nn] = 0.0;

			// energy flux
			ff[j][i].fn[nn] -= (NstiffFv(nn, i, j, xx) - NstiffFv(nn, i-1, j, xx)) /vip12 /(vip1-vi);

			// pitch angle flux
			ff[j][i].fn[nn] -= (NstiffFxi(nn, i, j, xx) - NstiffFxi(nn, i, j-1, xx)) /(xijp1-xij);

		}
	}
};

/*
 * Prepare a vec for integral
 */
template <class knockon_type>
void fk_equ_slab<knockon_type>::PrepareInt(Field **x_int, PetscBool ComputeJ)
{
	PetscInt i, j;
    PetscScalar dxi, dv;

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi
		xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);
		xijp12 = my_mesh->yy[j];
		xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j]) : (1.0);
        dxi = xijp1 - xij;
		for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
			vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
			vip12 = my_mesh->xx[i];
			vip1 = my_mesh->xf[i];
            dv = vip1 - vi;
            //if (vip12<=p_cutoff && j==1) PetscPrintf(PETSC_COMM_SELF,"vip12 = %g, p_cutoff=%g\n",vip12, p_cutoff);
            if (vip12>p_cutoff){
                if(ComputeJ){
                    // 2*PI*p^2/gamma/me*(f*p*xi)
                    x_int[j][i].fn[0] *= 2.0*PI*vip12*vip12/sqrt(1+vip12*vip12)*xijp12*dxi*dv; 
                }
                else{
                    // 2*PI*p^2
                    x_int[j][i].fn[0] *= 2.0*PI*vip12*dxi*dv; 
                }
            }
            else{
                x_int[j][i].fn[0] = 0.0;
            }
		}
	}
};

/*
 * Compute max dt for the advection term
 * note the local time step is determined by 
 *      dtlocal = CFL/( fabs(speedFv/v)/dv + fabs(speedFxi)/dxi )
 */
template <class knockon_type>
PetscScalar fk_equ_slab<knockon_type>::EvalMaxdt()
{
	PetscInt    i, j;
    double dt=1e3, dtlocal, CFL=0.9;
    PetscScalar aTv_, bTv_, speedFv, speedFxi;
    pstate      state=E_Field->get_state();

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);
		xijp12 = my_mesh->yy[j];
		xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j]) : (1.0);

		for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v

			// velocities evaluated at j, at j+1 and j+1/2.
			vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
			vip12 = my_mesh->xx[i];

            // compute speed for Fv (copied from NstiffFv function)
	        if (i == -1 ) {
		        vip1 = my_mesh->xmin;
            }
	        else {
		        vip1 = my_mesh->xf[i];
            }

	        aTv_ = aTv[i-my_mesh->xs+1];
	        bTv_ = bTv[i-my_mesh->xs+1];
    
	        speedFv = -vip1*( state.E*xijp12 + state.alpha*vip1*sqrt(1+vip1*vip1)*( (1. - xijp12*xijp12) + quad(rho)*vip1*vip1*quad(xijp12) ));
	        speedFv -= (bTv_*vip1 - aTv_);  

            // compute speed For Fxi (copied from NstiffFxi function)
            if (j == my_mesh->My - 1 || j==-1)  {
		        speedFxi = 0.0;
	        } else {
		        speedFxi = -(state.E/vip12) + state.alpha/sqrt(1+vip12*vip12)*xijp1;
            }

            dtlocal = 1./( fabs(speedFv/vip12)/(vip1-vi) + fabs(speedFxi)/(xijp1-xij) );

            dt = PetscMin(dt, dtlocal);
		}
	}

    double global_dt;
    MPI_Allreduce(&dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);

    return CFL*global_dt;
};




/* function to compute the xi flux on cell boundary*/
template <class knockon_type>
PetscScalar fk_equ_slab<knockon_type>::Fxi(PetscInt nn, int i, int j, Field **xx, PetscBool implicit)
{
	double      coef, f(0.0), flux;
	double      xijp1, yr, yu, yd;
    pstate      state=E_Field->get_state();

	if (j == my_mesh->My - 1 || j == -1)  {
		return 0.0;
	} else {
		xijp1 = my_mesh->yf[j];

       /*
        * Advection Part
        * The advection term for xi: (see RFP notes main.pdf eq. 16)
        *   Flux_xi = (1-xi^2)*coef*F where coef = -E/p + alpha*xi/gamma
        */
		coef = -(state.E/vip12) + state.alpha/sqrt(1+vip12*vip12)*xijp1;

        if (implicit)
        {
            //advection term
		    if (coef < 0) {
		    	if (j == my_mesh->My-2 )
		    		f = (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
		    	else {
		    		yr = my_mesh->yy[j+2]; yu = my_mesh->yy[j+1]; yd = my_mesh->yy[j];
		    		f = Uface(xx[j+2][i].fn[nn], xx[j+1][i].fn[nn], xx[j][i].fn[nn], yr, yu, yd, xijp1, diff_type);
		    	}
		    } else {
		    	if (j == 0)
		    		f = (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
		    	else {
		    		yr = my_mesh->yy[j-1]; yu = my_mesh->yy[j]; yd = my_mesh->yy[j+1];
		    		f = Uface(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], yr, yu, yd, xijp1, diff_type);
		    	}
		    }
        }
        else
        {
            f = 0.; //advection flux is defined in Nstiff part for IMEX
        }

        /*
         * Diffusion Part
         */

		// pitch-angle diffusion
        if (j == 0)
        {
            //note boundary still use low-order dfdx; high-order could be dangerous
		    flux = coef * f  - dTv[i-my_mesh->xs] * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                            my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j], my_mesh->yy[j+2]-my_mesh->yf[j]);
        }
		else if (j<my_mesh->My-2)
        {
            //used points are (j-1,i), (j,i), (j+1,i), (j+2,i); the order of the points is not important
            double yface=my_mesh->yf[j];
		    flux = coef * f  - dTv[i-my_mesh->xs] * Uface.dfdx( xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                    		               my_mesh->yy[j-1]-yface, my_mesh->yy[j]-yface, my_mesh->yy[j+1]-yface, my_mesh->yy[j+2]-yface) ;
        }                    
        else {
		    flux = coef * f  - dTv[i-my_mesh->xs] * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], 
                            my_mesh->yy[j-1]-my_mesh->yf[j], my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j]);
        }

        // quasilinear diffusion (new version)
        if (dw_ && true){   
			double D, fluxD;
            double yface=my_mesh->yf[j];

			int idx1 = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
			int idx2 = (i-my_mesh->xs+1) + (j+1-my_mesh->ys+1)*(my_mesh->xm+2);

			// d^2f/dxi^2
			D = 0.5*(Dw.xixi[idx1] + Dw.xixi[idx2]);

        	if (j == 0) {
			    fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j], my_mesh->yy[j+2]-my_mesh->yf[j]);
            }
		    else if (j<my_mesh->My-2) {
			    fluxD = -D/(vip12*vip12) * Uface.dfdx( xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        		my_mesh->yy[j-1]-yface, my_mesh->yy[j]-yface, my_mesh->yy[j+1]-yface, my_mesh->yy[j+2]-yface);
            }                    
            else {
			    fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], 
                                my_mesh->yy[j-1]-my_mesh->yf[j], my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j]);
            }

			// the off-diagonal coefficient
			D = 0.5*(Dw.pxi[idx1] + Dw.pxi[idx2]);

			// d^2f/dxidp term (a central scheme)
            // In numerical flux, we compute df/dp at xi=yf[j] and vi=xx[i]
			if (i > 0 && i < (my_mesh->Mx-1)) {
                 double fim1, fi, fip1, fip2;

                 //linear interpolation
                 fim1=(xx[j+1][i-1].fn[nn] * (yface-my_mesh->yy[j]) + xx[j][i-1].fn[nn] * (my_mesh->yy[j+1]-yface)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
                 fi=(xx[j+1][i  ].fn[nn] * (yface-my_mesh->yy[j]) + xx[j][i  ].fn[nn] * (my_mesh->yy[j+1]-yface)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
                 fip1=(xx[j+1][i+1].fn[nn] * (yface-my_mesh->yy[j]) + xx[j][i+1].fn[nn] * (my_mesh->yy[j+1]-yface)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
                 fip2=(xx[j+1][i+2].fn[nn] * (yface-my_mesh->yy[j]) + xx[j][i+2].fn[nn] * (my_mesh->yy[j+1]-yface)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);

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

		return (1 - xijp1*xijp1) * flux;
	}

};


/* function to compute velocity space flux (p or v) on cell boundary*/
template <class knockon_type>
PetscScalar fk_equ_slab<knockon_type>::Fv(PetscInt nn, int i, int j, Field **xx, PetscBool implicit)
{
	double  flux, f(0.0), coef;
	double  vip1, xr, xu, xd;
	double  aTv_ = aTv[i-my_mesh->xs+1];
	double  bTv_ = bTv[i-my_mesh->xs+1];
    pstate  state=E_Field->get_state();
    
	if (i == -1 ) {
		vip1 = my_mesh->xmin;
    }
	else {
		vip1 = my_mesh->xf[i];
    }

    /*
     * Advection part
     * The advection term for p: (see RFP notes main.pdf eq. 15)
     *   Flux_p = coef*F where coef = -p[E*xi + alpha*p*gamma*(1-xi^2)]-(p*C_F-C_A) 
     */

    //XXX the rho related term is removed in runaway2d-steady
	coef = -vip1*( state.E*xijp12 + state.alpha*vip1*sqrt(1+vip1*vip1)*( (1. - xijp12*xijp12) + quad(rho)*vip1*vip1*quad(xijp12) ));
	coef -= (bTv_*vip1 - aTv_);  //remember to change here if you want to solve f, pf, p^2f
    if (implicit) {

	    if (bc_type == 1) {  // zero flux
	    	if (coef < 0.) {
	    		if (i == my_mesh->Mx - 1) {
	    			f = 0.0;
	    		} else if (i == my_mesh->Mx - 2) {
	    			xr = my_mesh->xmax; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
	    			f = Uface(0., xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type); //Uface(xx[j][i+1].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn]); //xx[j][i+1].fn[nn]; //0.5*(xx[j][i].fn[nn] + xx[j][i+1].fn[nn]);
	    		} else if (i == -1) {
	    			xr = my_mesh->xx[1]; xu = my_mesh->xx[0]; xd = 2.*my_mesh->xmin - my_mesh->xx[0];
	    			f = Uface(xx[j][1].fn[nn], xx[j][0].fn[nn], (ra*Fmax(xd)*xd), xr, xu, xd, vip1, diff_type);
	    		} else {
	    			xr = my_mesh->xx[i+2]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
	    			f = Uface(xx[j][i+2].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type);
	    		}
	    	} else {
	    		if (i == my_mesh->Mx - 1) {
	    			f = 0.0; 
	    		} else if (i == 0) {
	    			xr = 2.*my_mesh->xmin - my_mesh->xx[0]; xu = my_mesh->xx[0]; xd = my_mesh->xx[1];
	    			f = Uface(((ra*Fmax(xr))*xr), xx[j][0].fn[nn], xx[j][1].fn[nn], xr, xu, xd, vip1, diff_type);
	    		} else if (i == -1) {
	    			xr = (4.*my_mesh->xmin - 3.*my_mesh->xx[0]); xu = (2.*my_mesh->xmin - my_mesh->xx[0]); xd = my_mesh->xx[0];
	    			f = Uface((ra*Fmax(xr)*xr), (ra*Fmax(xu)*xu), xx[j][0].fn[nn], xr, xu, xd, vip1, diff_type);
	    		} else {
	    			xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xx[i+1];
	    			f = Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xr, xu, xd, vip1, diff_type);
	    		}
	    	}
	    	flux = coef*f;

	    } 
        else {  // finite outgoing flux
	    	PetscPrintf(PETSC_COMM_WORLD,"======CHECK ME!======\n");
	    	if (coef < 0.) {
	    		if (i == my_mesh->Mx - 1) {
	    			f = 0.; //Uface(0.0, 0.0, xx[j][i].fn[nn]);//(3.0/8.0)*xx[j][i].fn[nn];
	    		} else if (i == my_mesh->Mx - 2) {
	    			xr = my_mesh->xmax; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
	    			f = Uface(xx[j][i+1].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type); //Uface(0.0, xx[j][i+1].fn[nn], xx[j][i].fn[nn]); 
	    		} else if (i == -1) {
	    			xr = my_mesh->xx[1]; xu = my_mesh->xx[0]; xd = 2.*my_mesh->xmin - my_mesh->xx[0];
	    			f = Uface(xx[j][1].fn[nn], xx[j][0].fn[nn], (ra*Fmax(xd)*xd), xr, xu, xd, vip1, diff_type);
	    		} else {
	    			xr = my_mesh->xx[i+2]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
	    			f = Uface(xx[j][i+2].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type);
	    		}
	    	} else {
	    		if (i == (my_mesh->Mx - 1)) {
	    			xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xmax;
	    			f = xx[j][i].fn[nn]; 
	    		} else if (i == 0) {
	    			xr = 2.*my_mesh->xmin - my_mesh->xx[0]; xu = my_mesh->xx[0]; xd = my_mesh->xx[1];
	    			f = Uface(((ra*Fmax(xr))*xr), xx[j][0].fn[nn], xx[j][1].fn[nn], xr, xu, xd, vip1, diff_type);
	    		} else if (i == -1) {
	    			xr = (4.*my_mesh->xmin - 3.*my_mesh->xx[0]); xu = (2.*my_mesh->xmin - my_mesh->xx[0]); xd = my_mesh->xx[0];
	    			f = Uface((ra*Fmax(xr)*xr), (ra*Fmax(xu)*xu*xu), xx[j][0].fn[nn], xr, xu, xd, vip1, diff_type);
	    		} else {
	    			xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xx[i+1];
	    			f = Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xr, xu, xd, vip1, diff_type);
	    		}
	    	}
	    	flux = coef*(f);
	    }

    }
    else
    {
        flux=0.;    //advection flux is defined in explicit part for IMEX
    }

    /* 
     * Diffusion part
     */

	// energy diffusive flux due to collision (central differencing)
	if (i == -1) {
		xd = (2.*my_mesh->xmin - my_mesh->xx[0]);
		flux -= vip1*aTv_ * (xx[j][0].fn[nn] - (ra*Fmax(xd)*xd))/(my_mesh->xx[0] - xd) ;
	} 
    else if (i == 0) {
		flux -= vip1*aTv_ * Uface.dfdx(xx[j][i].fn[nn], xx[j][i+1].fn[nn], xx[j][i+2].fn[nn], my_mesh->xx[i]-my_mesh->xf[i], my_mesh->xx[i+1]-my_mesh->xf[i], my_mesh->xx[i+2]-my_mesh->xf[i]);
	} 
    else if (i < (my_mesh->Mx-1)) {

        double xface=my_mesh->xf[i];
		flux -= vip1*aTv_ * Uface.dfdx(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xx[j][i+2].fn[nn],
                my_mesh->xx[i-1]-xface, my_mesh->xx[i]-xface, my_mesh->xx[i+1]-xface, my_mesh->xx[i+2]-xface);

		// with quasilinear diffusion
        if (dw_ && true){
		    // quasilinear diffusion (new version)
		    double Dpp, Dpxi, fluxD;
            double xface=my_mesh->xf[i];
			int idx1 = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
			int idx2 = (i+1-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);

			Dpp = 0.5*(Dw.pp[idx1] + Dw.pp[idx2]) * (1.-xijp12*xijp12);

		    // d^2/dp^2 term
		    fluxD = -vip1*Dpp * Uface.dfdx(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xx[j][i+2].fn[nn],
                        my_mesh->xx[i-1]-xface, my_mesh->xx[i]-xface, my_mesh->xx[i+1]-xface, my_mesh->xx[i+2]-xface);

		    // the off-diagonal coefficient
			Dpxi = 0.5*(Dw.pxi[idx1] + Dw.pxi[idx2])*(1.-xijp12*xijp12);

		    // d^2/dpdxi term (central scheme)
            // In the numerical flux, we compute df/dxi at vi=xf[i] xi=yy[j]
		    if (j>0 && j<my_mesh->My-2) {
                double fim1, fi, fip1, fip2;

                //linear interpolation
                fim1=(xx[j-1][i+1].fn[nn] * (xface-my_mesh->xx[i]) + xx[j-1][i].fn[nn] * (my_mesh->xx[i+1]-xface)) / (my_mesh->xx[i+1]-my_mesh->xx[i]);
                  fi=(xx[j  ][i+1].fn[nn] * (xface-my_mesh->xx[i]) + xx[j  ][i].fn[nn] * (my_mesh->xx[i+1]-xface)) / (my_mesh->xx[i+1]-my_mesh->xx[i]);
                fip1=(xx[j+1][i+1].fn[nn] * (xface-my_mesh->xx[i]) + xx[j+1][i].fn[nn] * (my_mesh->xx[i+1]-xface)) / (my_mesh->xx[i+1]-my_mesh->xx[i]);
                fip2=(xx[j+2][i+1].fn[nn] * (xface-my_mesh->xx[i]) + xx[j+2][i].fn[nn] * (my_mesh->xx[i+1]-xface)) / (my_mesh->xx[i+1]-my_mesh->xx[i]);

		         flux += fluxD - Dpxi*Uface.dfdx(fim1, fi, fip1, fip2, 
                                         my_mesh->yy[j-1]-my_mesh->yy[j], 0.0, my_mesh->yy[j+1]-my_mesh->yy[j], my_mesh->yy[j+2]-my_mesh->yy[j]);
            }	
            else if (j==0 || j==my_mesh->My-2) { 
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
		    } 
            else {
		    	if (i == my_mesh->Mx - 1 || i <= 0) {
			        flux += Dpp * xx[j][i].fn[nn];
		    	} else {
		    		xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xx[i+1];
		    		flux += Dpp * Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xr, xu, xd, vip1, diff_type);
		    	}
		    }
		}

	}
	return flux;
};

/* function to compute the xi flux on cell boundary*/
template <class knockon_type>
PetscScalar fk_equ_slab<knockon_type>::NstiffFxi(PetscInt nn, int i, int j, Field **xx)
{
	double coef, f(0.0), flux;
	double xijp1, yr, yu, yd;
    pstate      state=E_Field->get_state();

    /*
     * The advection term for xi: (see RFP notes main.pdf eq. 16)
     *   Flux_xi = (1-xi^2)*coef*F where coef = -E/p + alpha*xi/gamma
     */

	if (j == my_mesh->My - 1 || j == -1)  {
		return 0.0;
	} 
    else {
		xijp1 = my_mesh->yf[j];
		coef = -(state.E/vip12) + state.alpha/sqrt(1+vip12*vip12)*xijp1;

		if (coef < 0) {
			if (j == my_mesh->My-2 )
				f = (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
			else {
				yr = my_mesh->yy[j+2]; yu = my_mesh->yy[j+1]; yd = my_mesh->yy[j];
				f = Uface(xx[j+2][i].fn[nn], xx[j+1][i].fn[nn], xx[j][i].fn[nn], yr, yu, yd, xijp1, diff_type);
			}
		} 
        else {
			if (j == 0)
				f = (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) / (my_mesh->yy[j+1]-my_mesh->yy[j]);
			else {
				yr = my_mesh->yy[j-1]; yu = my_mesh->yy[j]; yd = my_mesh->yy[j+1];
				f = Uface(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], yr, yu, yd, xijp1, diff_type);
			}
		}

        flux = coef*f;

		return (1 - xijp1*xijp1) * flux;
	}

};

/* function to compute velocity space flux on cell boundary*/
template <class knockon_type>
PetscScalar fk_equ_slab<knockon_type>::NstiffFv(PetscInt nn, int i, int j, Field **xx)
{
	double flux, f(0.0), coef, vip1;
    pstate      state=E_Field->get_state();

	if (i == -1 )
		vip1 = my_mesh->xmin;
	else
		vip1 = my_mesh->xf[i];

	double xr, xu, xd;
	double aTv_ = aTv[i-my_mesh->xs+1];
	double bTv_ = bTv[i-my_mesh->xs+1];

    /*
     * Advection part
     * The advection term for p: (see RFP notes main.pdf eq. 15)
     *   Flux_p = coef*F where coef = -p[E*xi + alpha*p*gamma*(1-xi^2)]-(p*C_F-C_A) 
     */
	coef = -vip1*( state.E*xijp12 + state.alpha*vip1*sqrt(1+vip1*vip1)*( (1. - xijp12*xijp12) + quad(rho)*vip1*vip1*quad(xijp12) ));
	coef -= (bTv_*vip1 - aTv_);  

	if (bc_type == 1) {  // zero flux
		if (coef < 0.) {
			if (i == my_mesh->Mx - 1) {
				f = 0.0;
			} else if (i == my_mesh->Mx - 2) {
				xr = my_mesh->xmax; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
				f = Uface(0., xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type); //Uface(xx[j][i+1].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn]); //xx[j][i+1].fn[nn]; //0.5*(xx[j][i].fn[nn] + xx[j][i+1].fn[nn]);
			} else if (i == -1) {
				xr = my_mesh->xx[1]; xu = my_mesh->xx[0]; xd = 2.*my_mesh->xmin - my_mesh->xx[0];
				f = Uface(xx[j][1].fn[nn], xx[j][0].fn[nn], (ra*Fmax(xd)*xd), xr, xu, xd, vip1, diff_type);
			} else {
				xr = my_mesh->xx[i+2]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
				f = Uface(xx[j][i+2].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type);
			}
		} 
        else {
			if (i == my_mesh->Mx - 1) {
				f = 0.0; 
			} else if (i == 0) {
				xr = 2.*my_mesh->xmin - my_mesh->xx[0]; xu = my_mesh->xx[0]; xd = my_mesh->xx[1];
				f = Uface(((ra*Fmax(xr))*xr), xx[j][0].fn[nn], xx[j][1].fn[nn], xr, xu, xd, vip1, diff_type);
			} else if (i == -1) {
				xr = (4.*my_mesh->xmin - 3.*my_mesh->xx[0]); xu = (2.*my_mesh->xmin - my_mesh->xx[0]); xd = my_mesh->xx[0];
				f = Uface((ra*Fmax(xr)*xr), (ra*Fmax(xu)*xu), xx[j][0].fn[nn], xr, xu, xd, vip1, diff_type);
			} else {
				xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xx[i+1];
				f = Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xr, xu, xd, vip1, diff_type);
			}
		}
		flux = coef*f;

	}
    else {  // finite outgoing flux
		PetscPrintf(PETSC_COMM_WORLD,"======CHECK ME!======\n");
		if (coef < 0.) {
			if (i == my_mesh->Mx - 1) {
				f = 0.; //Uface(0.0, 0.0, xx[j][i].fn[nn]);//(3.0/8.0)*xx[j][i].fn[nn];
			} else if (i == my_mesh->Mx - 2) {
				xr = my_mesh->xmax; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
				f = Uface(xx[j][i+1].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type); 
			} else if (i == -1) {
				xr = my_mesh->xx[1]; xu = my_mesh->xx[0]; xd = 2.*my_mesh->xmin - my_mesh->xx[0];
				f = Uface(xx[j][1].fn[nn], xx[j][0].fn[nn], (ra*Fmax(xd)*xd), xr, xu, xd, vip1, diff_type);
			} else {
				xr = my_mesh->xx[i+2]; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
				f = Uface(xx[j][i+2].fn[nn], xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type);
			}
		} 
        else {
			if (i == (my_mesh->Mx - 1)) {
				xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xmax;
				f = xx[j][i].fn[nn]; 
			} else if (i == 0) {
				xr = 2.*my_mesh->xmin - my_mesh->xx[0]; xu = my_mesh->xx[0]; xd = my_mesh->xx[1];
				f = Uface(((ra*Fmax(xr))*xr), xx[j][0].fn[nn], xx[j][1].fn[nn], xr, xu, xd, vip1, diff_type);
			} else if (i == -1) {
				xr = (4.*my_mesh->xmin - 3.*my_mesh->xx[0]); xu = (2.*my_mesh->xmin - my_mesh->xx[0]); xd = my_mesh->xx[0];
				f = Uface((ra*Fmax(xr)*xr), (ra*Fmax(xu)*xu*xu), xx[j][0].fn[nn], xr, xu, xd, vip1, diff_type);
			} else {
				xr = my_mesh->xx[i-1]; xu = my_mesh->xx[i]; xd = my_mesh->xx[i+1];
				f = Uface(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xr, xu, xd, vip1, diff_type);
			}
		}
		flux = coef*f;
	}

	return flux;
};



// this is again broken -QT
/* Set pre-conditioner for the implicit part (dF/dX + a dF/dXdot) of the IMEX scheme*/
// a stable schme is obatined by putting all diffusion terms implicitly and all convective terms explicitly
// so the jacobian can be simplified to take care of diffusion terms only
template <class knockon_type>
void fk_equ_slab<knockon_type>::SetIJacobian(Field **x, Field **xdot, PetscReal a, Mat jac, AppCtx *user)
{
	PetscErrorCode ierr;
	int         	i,j,k;
	MatStencil       col[7],row;
	PetscScalar      v[7];

	PetscReal        aTv_, bTv_, cTv_, dTv_, aTxi, cTxi, E, temp;
	PetscReal        av,bv,cv, bs, dv,ev,fv;
	PetscReal        att,btt,ctt, afs,bfs,cfs, axi,bxi,cxi;

    pstate      state=E_Field->get_state();
	E = state.E;

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) {

		xij = (j>0) ? (0.5*(my_mesh->yy[j-1]+my_mesh->yy[j])) : (-1.0);
		xijp12 = my_mesh->yy[j];
		xijp1 = (j<(my_mesh->My-1)) ? (0.5*(my_mesh->yy[j]+my_mesh->yy[j+1])) : (1.0);

		for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) {

			row.k = k; row.j = j; row.i = i;

			// velocities evaluated at j, at j+1 and j+1/2.
			vi = (i==0) ? (my_mesh->xmin) : (0.5*(my_mesh->xx[i-1] + my_mesh->xx[i]));
			vip12 = my_mesh->xx[i];
			vip1 = (i<(my_mesh->Mx-1)) ? (0.5*(my_mesh->xx[i] + my_mesh->xx[i+1])) : (my_mesh->xmax);

			// collision related coefficients
			aTv_ = sqrt(1+vip1*vip1)/vip12*Chand(vip1) /square1(vip1-vi);
			cTv_ = sqrt(1+vi*vi)/vip12*Chand(vi) /square1(vip1-vi);

			dTv_ = dTv[i-my_mesh->xs] /square1(xijp1-xij);
			aTxi = (1-xijp1*xijp1)*dTv_;
			cTxi = (1-xij*xij)*dTv_;

			// set Jacobian elements for test-particle collision operators
			if (i == 0 ) {
				v[0] = 1.0;	                        col[0].k = k;  col[0].j = j;   col[0].i = i;
				ierr = MatSetValuesStencil(jac,1,&row,1,col,v,INSERT_VALUES);

			} else {
				if (i == my_mesh->Mx-1) {
					if (bc_type == 1) {
						v[0] = a + cTv_;			col[0].j = j;   col[0].i = i;
					} else {
						v[0] = a + aTv_ + bTv_; 	col[0].j = j;   col[0].i = i;
					}
					v[1] = -cTv_;			    col[1].j = j;   col[1].i = i-1;

					if (j == my_mesh->My-1) {
						v[0] += cTxi;
						v[2] = -cTxi;    	  	col[2].j = j-1;  col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);//CHKERRQ(ierr);
					} else if (j == 0) {
						v[0] += aTxi;
						v[2] = -aTxi; 	    	col[2].j = j+1; col[2].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);//CHKERRQ(ierr);
					} else {
						v[2] = -aTxi;     	   col[2].j = j+1;   col[2].i = i;
						v[0] += (aTxi+cTxi);
						v[3] = -cTxi;     	   col[3].j = j-1;   col[3].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);//CHKERRQ(ierr);
					}
				} else {
					v[0] = -aTv_;    			col[0].j = j;   col[0].i = i+1;
					v[1] = a + aTv_+cTv_;		col[1].j = j;   col[1].i = i;
					v[2] = -cTv_;  	   			col[2].j = j;   col[2].i = i-1;

					if (j == my_mesh->My-1) {
						v[1] += cTxi;
						v[3] = -cTxi;      	col[3].j = j-1;  col[3].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);//CHKERRQ(ierr);
					} else if (j == 0) {
						v[1] += aTxi;
						v[3] = -aTxi;     	   col[3].j = j+1; col[3].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);//CHKERRQ(ierr);
					} else {
						v[3] = -aTxi;     	   col[3].j = j+1;   col[3].i = i;
						v[1] += (aTxi+cTxi);
						v[4] = -cTxi;     	   col[4].j = j-1;   col[4].i = i;

						ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);//CHKERRQ(ierr);
					}
				}
			}
		}
	}
	// pitch angle flux due to electric field and mirror force
	////				temp = -0.5*( E/vip12 + 0.5*ctr/q*asp * vip12/sqrt(1+vip12*vip12) * ra*sin(thetakp12) ) /hy;
	//				axi = 0;//(1 - xijp1*xijp1) *temp;
	////				bxi = (xij*xij - xijp1*xijp1) *temp;
	//				cxi = 0;//-(1 - xij*xij) *temp;
	//				// radiation induced pitch-angle flux
	////				temp = 0.5*sr/sqrt(1+vip12*vip12) /hy;
	////				axi += xijp1*(1 - xijp1*xijp1) *temp;
	////				cxi -= xij*(1 - xij*xij) *temp;
	//
	//				/* energy flux */
	//				// electric field caused flux
	//				temp = -0.5 * E*xijp12 /square(vip12)/hx;
	//				av = Env(vip1)*vip1*vip1 *temp;
	//				bv = (Env(vip1)*vip1*vip1 - Env(vi)*vi*vi) *temp;
	//				cv = -Env(vi)*vi*vi *temp;
	//				// radiation caused flux
	//				temp = -0.5*sr*(1-xijp12*xijp12) /square(vip12)/hx;
	//				av += Env(vip1)*cubic(vip1)*sqrt(1+vip1*vip1) *temp;
	//				bv += (Env(vip1)*cubic(vip1)*sqrt(1+vip1*vip1) - Env(vi)*cubic(vi)*sqrt(1+vi*vi)) *temp;
	//				cv -= Env(vi)*cubic(vi)*sqrt(1+vi*vi) *temp;
	//				temp = -0.5*sr*quad(rho)*quad(xijp12) /square(vip12)/hx;
	//				av += Env(vip1)*cubic(vip1)*cubic(vip1)*sqrt(1+vip1*vip1) *temp;
	//				bv += (Env(vip1)*cubic(vip1)*cubic(vip1)*sqrt(1+vip1*vip1) - Env(vi)*cubic(vi)*cubic(vi)*sqrt(1+vi*vi)) *temp;
	//				cv -= Env(vi)*cubic(vi)*cubic(vi)*sqrt(1+vi*vi) *temp;

	//		if (i == 0 ) {
	//			v[0] = 1.0;	                        col[0].k = k;  col[0].j = j;   col[0].i = i;
	//			ierr = MatSetValuesStencil(jac,1,&row,1,col,v,INSERT_VALUES);
	//
	//		} else {
	//			if (i == my_mesh->Mx-1) {
	//				if (bc_type == 1) {
	//					v[0] = a - bTv_ + cv;			col[0].j = j;   col[0].i = i;
	//				} else {
	//					v[0] = a - bTv_ + bv;			col[0].j = j;   col[0].i = i;
	//				}
	//				v[1] = -cTv_ + cv;			    col[1].j = j;   col[1].i = i-1;
	//
	//				if (j == my_mesh->My-1) {
	//					v[0] += -cTxi;
	//					v[2] = cTxi;    	  	col[2].j = j-1;  col[2].i = i;
	//
	//					ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);//CHKERRQ(ierr);
	//				} else if (j == 0) {
	//					v[0] += -aTxi;
	//					v[2] = aTxi; 	    	col[2].j = j+1; col[2].i = i;
	//
	//					ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);//CHKERRQ(ierr);
	//				} else {
	//					v[2] = aTxi;     	   col[2].j = j+1;   col[2].i = i;
	//					v[0] += -(aTxi+cTxi);
	//					v[3] = cTxi;     	   col[3].j = j-1;   col[3].i = i;
	//
	//					ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);//CHKERRQ(ierr);
	//				}
	//			} else {
	//				v[0] = -aTv_ + av;    			col[0].j = j;   col[0].i = i+1;
	//				v[1] = a -bTv_ + bv;			col[1].j = j;   col[1].i = i;
	//				v[2] = -cTv_ + cv;  	   			col[2].j = j;   col[2].i = i-1;
	//
	//				if (j == my_mesh->My-1) {
	//					v[1] += -cTxi;
	//					v[3] = cTxi;      	col[3].j = j-1;  col[3].i = i;
	//
	//					ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);//CHKERRQ(ierr);
	//				} else if (j == 0) {
	//					v[1] += -aTxi;
	//					v[3] = aTxi;     	   col[3].j = j+1; col[3].i = i;
	//
	//					ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);//CHKERRQ(ierr);
	//				} else {
	//					v[3] = aTxi;     	   col[3].j = j+1;   col[3].i = i;
	//					v[1] += -(aTxi+cTxi);
	//					v[4] = cTxi;     	   col[4].j = j-1;   col[4].i = i;
	//
	//					ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);//CHKERRQ(ierr);
	//				}
	//			}
	//		}
	//	}

};

template class fk_equ_slab<knockon_rp>;
template class fk_equ_slab<knockon_chiu>;
template class fk_equ_slab<knockon_none>;
