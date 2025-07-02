
/*
 * fk_equ_slab.cpp
 *
 *  Created on: Sep 25, 2014
 *      Author: zehuag
 */

#include "fk_equ_slab.h"

/* constructor
 * reading parameters and set up the collision operator
 * */
fk_equ_slab::fk_equ_slab(mesh *mesh_, char* param_file) : 
fk_equ(mesh_, param_file), Dw(mesh_)
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
void fk_equ_slab::initialize(Field **xx, AppCtx *user)
{
	int i, j, nn;
	PetscReal v,xi;

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
void fk_equ_slab::Eval(Field **xx, Field **ff, AppCtx *user)
{
	PetscInt i, j, k; 
	PetscInt   nn = 0;
	PetscReal   aTv_,bTv_,cTv_, dTv_;

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);
		xijp12 = my_mesh->yy[j];
		xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j]) : (1.0);

		for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
			// velocities evaluated at j, at j+1 and j+1/2.
			vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
			vip12 = my_mesh->xx[i];
			vip1 = my_mesh->xf[i];

			// no time derivative and no knockon source term
			// energy flux
			ff[j][i].fn[nn] = (Fv(nn, i, j, xx) - Fv(nn, i-1, j, xx))/vip12 /(vip1-vi);  //solving v*f

			// pitch angle flux
			ff[j][i].fn[nn] += (Fxi(nn, i, j, xx) - Fxi(nn, i, j-1, xx)) /(xijp1-xij);

		}
	}
};

/* function to compute the xi flux on cell boundary
 * Here we evaluate Fxi(i,j)=Fxi_{i,j+1/2} at xi=yf[j]
 */
PetscScalar fk_equ_slab::Fxi(PetscInt nn, int i, int j, Field **xx)
{
	PetscScalar coef, f(0.0), E, flux;

	PetscScalar xijp1; 
	PetscScalar yr, yu, yd;

	if (j == my_mesh->My - 1) {
		return 0.0;
	} else if (j == -1) {
		return 0.0;
	} 
        else {
		xijp1 = my_mesh->yf[j];
		E = Efield(ra);
		coef = -(E/vip12) + sr/sqrt(1+vip12*vip12)*xijp1;

		if (coef < 0.) {
			if (j == my_mesh->My-2 )
			    f = (xx[j+1][i].fn[nn] * (xijp1-my_mesh->yy[j]) + xx[j][i].fn[nn] * (my_mesh->yy[j+1]-xijp1)) 
                                / (my_mesh->yy[j+1]-my_mesh->yy[j]);
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

	    // collisional pitch-angle diffusion
        if (true) //high-order version of diffusion
        {
        	if (j == 0)
                //boundary still use low-order dfdx; high-order could be dangerous
			    flux = coef * f  - dTv[i-my_mesh->xs] * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                                my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j], my_mesh->yy[j+2]-my_mesh->yf[j]);
		    else if (j<my_mesh->My-2)
            {
                //used points are (j-1,i), (j,i), (j+1,i), (j+2,i); the order of the points is not important
                double yface=my_mesh->yf[j];
			    flux = coef * f  - dTv[i-my_mesh->xs] * Uface.dfdx( xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        		               my_mesh->yy[j-1]-yface, my_mesh->yy[j]-yface, my_mesh->yy[j+1]-yface, my_mesh->yy[j+2]-yface) ;
            }                    
            else
			    flux = coef * f  - dTv[i-my_mesh->xs] * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], 
                                my_mesh->yy[j-1]-my_mesh->yf[j], my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j]);
        }
        else {
                    //lower-order version of diffusion
		    if (j == 0)
			flux = coef * f  - dTv[i-my_mesh->xs] * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j], my_mesh->yy[j+2]-my_mesh->yf[j]);
		    else
			flux = coef * f  - dTv[i-my_mesh->xs] * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], my_mesh->yy[j-1]-my_mesh->yf[j], my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j]);
        }

        if (dw_ && true){   
            // with quasilinear diffusion (new version)
			double D, fluxD;
            double yface=my_mesh->yf[j];

			int idx1 = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);
			int idx2 = (i-my_mesh->xs+1) + (j+1-my_mesh->ys+1)*(my_mesh->xm+2);

			// d^2f/dxi^2
			D = 0.5*(Dw.xixi[idx1] + Dw.xixi[idx2]);

        	if (j == 0)
			        fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                                my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j], my_mesh->yy[j+2]-my_mesh->yf[j]);
		    else if (j<my_mesh->My-2) {
			        fluxD = -D/(vip12*vip12) * 
                                        Uface.dfdx( xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], xx[j+2][i].fn[nn], 
                        		my_mesh->yy[j-1]-yface, my_mesh->yy[j]-yface, my_mesh->yy[j+1]-yface, my_mesh->yy[j+2]-yface);
            }                    
            else
			        fluxD = -D/(vip12*vip12) * Uface.dfdx(xx[j-1][i].fn[nn], xx[j][i].fn[nn], xx[j+1][i].fn[nn], 
                                my_mesh->yy[j-1]-my_mesh->yf[j], my_mesh->yy[j]-my_mesh->yf[j], my_mesh->yy[j+1]-my_mesh->yf[j]);

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


/* function to compute velocity space flux on cell boundary
 * Here we solve Fv(i,j)=Fv_{i+1/2,j}
 */
PetscScalar fk_equ_slab::Fv(PetscInt nn, int i, int j, Field **xx)
{
	PetscScalar flux, E, f(0.0), coef;

	E = Efield(ra); 
	PetscScalar vip1;

	if (i == -1 )
    {
	    vip1 = my_mesh->xmin;
    }
	else
    {
	    vip1 = my_mesh->xf[i];
    }

	PetscScalar xr, xu, xd;

	PetscScalar aTv_ = aTv[i-my_mesh->xs+1]; 	
    PetscScalar bTv_ = bTv[i-my_mesh->xs+1];

	coef = -vip1*( E*xijp12 + sr*vip1*sqrt(1+vip1*vip1)* (1. - xijp12*xijp12) );
	coef -= (bTv_*vip1 - aTv_);  //remember to change here corresponding for solving f, pf, p^2f

	if (bc_type == 1) {  // zero flux
		if (coef < 0.) {
			if (i == my_mesh->Mx - 1) {
				f = 0.0; 
			} else if (i == my_mesh->Mx - 2) {
				xr = my_mesh->xmax; xu = my_mesh->xx[i+1]; xd = my_mesh->xx[i];
				f = Uface(0., xx[j][i+1].fn[nn], xx[j][i].fn[nn], xr, xu, xd, vip1, diff_type); 
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

	// energy diffusive flux due to Coulomb collision (central differencing)
	if (i == -1) 
        {
		xd = (2.*my_mesh->xmin - my_mesh->xx[0]);
		flux -= vip1*aTv_ * (xx[j][0].fn[nn] - (ra*Fmax(xd)*xd))/(my_mesh->xx[0] - xd) ;
	} 
    else if (i == 0) {
		flux -= vip1*aTv_ * Uface.dfdx(xx[j][i].fn[nn], xx[j][i+1].fn[nn], xx[j][i+2].fn[nn], 
                my_mesh->xx[i]-my_mesh->xf[i], my_mesh->xx[i+1]-my_mesh->xf[i], my_mesh->xx[i+2]-my_mesh->xf[i]);
	} else if (i < (my_mesh->Mx-1)) {
           if (true)
           {
               double xface=my_mesh->xf[i];
		       flux -= vip1*aTv_ * Uface.dfdx(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], xx[j][i+2].fn[nn],
                       my_mesh->xx[i-1]-xface, my_mesh->xx[i]-xface, my_mesh->xx[i+1]-xface, my_mesh->xx[i+2]-xface);
           }
           else 
           {   //this is the low-order dfdx
		       flux -= vip1*aTv_ * Uface.dfdx(xx[j][i-1].fn[nn], xx[j][i].fn[nn], xx[j][i+1].fn[nn], 
                       my_mesh->xx[i-1]-my_mesh->xf[i], my_mesh->xx[i]-my_mesh->xf[i], my_mesh->xx[i+1]-my_mesh->xf[i]);
           }

		    
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
		        } else {
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


// this is broken -QT
/* Set pre-conditioner for the implicit part (dF/dX + a dF/dXdot) of the IMEX scheme*/
// a stable schme is obatined by putting all diffusion terms implicitly and all convective terms explicitly
// so the jacobian can be simplified to take care of diffusion terms only
void fk_equ_slab::SetJacobian(Field **x, Mat jac, AppCtx *user)
{
	PetscErrorCode ierr;
	int         	i,j,k;

	MatStencil       col[7],row;
	PetscScalar      v[7];

	PetscReal        aTv_, bTv_, cTv_, dTv_, aTxi, cTxi, E, temp;
	PetscReal        av,bv,cv, bs, dv,ev,fv;
	PetscReal        att,btt,ctt, afs,bfs,cfs, axi,bxi,cxi;

	E = Efield(ra);

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
		}
	}

};
