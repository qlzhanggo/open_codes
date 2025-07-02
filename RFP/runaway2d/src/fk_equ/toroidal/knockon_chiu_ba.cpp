/*
 * knockon_chiu_ba.cpp
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */
#include "knockon_chiu_ba.h"
#include <random>


knockon_chiu_ba::knockon_chiu_ba(mesh *mesh_, BACtx *ba_ctx_, double beta_, double pc_) : my_mesh(mesh_), ba_ctx(ba_ctx_), beta(beta_),
pc(pc_), ntheta(50), dtheta(PI/(ntheta-1)), dist(0.,1.)
{
	float xic = ba_ctx->xic;
	eps = xic*xic/(2.-xic*xic);

	PetscPrintf(PETSC_COMM_WORLD,"Using bounce-averaged Chiu knock-on source:\n");
	PetscPrintf(PETSC_COMM_WORLD,"beta = %lg\n", (beta));
	PetscPrintf(PETSC_COMM_WORLD,"pc = %lg\n", (pc));
	PetscPrintf(PETSC_COMM_WORLD,"eps = %lg\n", (eps));

	fv.resize(my_mesh->Mx * ntheta, 0.);
	src.resize(my_mesh->xm * my_mesh->My2, 0.);

	int color = (int)(my_mesh->xs/my_mesh->xm);
	int key   = (int)(my_mesh->ys/my_mesh->ym);

	MPI_Comm_split(PETSC_COMM_WORLD, color, key, &commy); //communicator in xi direction
	MPI_Comm_split(PETSC_COMM_WORLD, key, color, &commx); //communicator in vp direction

	int rank, size;
	MPI_Comm_rank(commx, &rank);
	MPI_Comm_size(commx, &size);

	// recv_count.resize(size);
	recv_count = std::unique_ptr<int[]>(new int[size]);
	displs = std::unique_ptr<int[]>(new int[size]);

	int send_count = my_mesh->xm;
	MPI_Allgather(&send_count, 1, MPI_INT, recv_count.get(), 1, MPI_INT, commx);
	MPI_Barrier(commx);

	// displs.resize(size);
	displs[0] = 0;
	for (int i=1; i<size; i++) {
		displs[i] = displs[i-1] + recv_count[i];
	}

	temp = std::unique_ptr<double[]>(new double[my_mesh->xm]);
	temp1 = std::unique_ptr<double[]>(new double[my_mesh->xm]);

	// for n=4 Gauss-quadrature rule
//	a[0] = -sqrt(3./7. + 2./7.*sqrt(6./5.)); a[1] = -sqrt(3./7. - 2./7.*sqrt(6./5.)); a[2] = -a[1]; a[3] = -a[0];
//	w[0] = 0.5 - sqrt(30.)/36.; w[1] = 0.5 + sqrt(30.)/36.; w[2] = w[1]; w[3] = w[0];
	// n=5
	a[0] = -sqrt(5. + 2.*sqrt(10./7.))/3.0; a[1] = -sqrt(5. - 2.*sqrt(10./7.))/3.0; a[2] = 0.; a[3] = -a[1]; a[4] = -a[0];
	w[0] = (322. - 13.*sqrt(70))/900.; w[1] = (322. + 13.*sqrt(70))/900.; w[2] = 128./225.; w[3] = w[1]; w[4] = w[0];

	// n=3
	av[0] = -sqrt(0.6); av[1] = 0; av[2] = -av[0];
	wv[0] = 5./9.; wv[1] = 8.0/9.0; wv[2] = wv[0];

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	gen = std::mt19937(rd());
};


/* perform poloidal integration of the source: v, xi momentum given at theta=0*/
PetscReal knockon_chiu_ba::Eval(const double &v, const double &xi)
{
	PetscReal gm=sqrt(v*v+1.), xi2, gm1, v1, theta0;

	if (xi < -ba_ctx->xic || xi > ba_ctx->xic)
		theta0 = PI;
	else {
		double kappa = 1. + (xi*xi/(ba_ctx->xic*ba_ctx->xic) - 1.)/(1. - xi*xi);
		theta0 = 2.*asin(kappa); // largest theta the trapped electron can reach
	}

	double sum = 0.;
	int ntheta = 100;
	float dtheta0 = theta0/(ntheta-1), theta;

	for (int idx=0; idx<(ntheta-1); idx++) // loop through theta range
	{
		theta = (idx+0.5)*dtheta0;

		/* computing the primary electron energy */
		xi2 = 1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xi*xi); // secondary electron pitch-angle at theta
		gm1 = 1. + 2./(xi2*(gm+1.)/(gm-1.) - 1.); // primary energy at theta

		if (gm1 > 1.) { // below rp curve
			v1 = sqrt(gm1*gm1 - 1.);

			if (gm1 > (2.*gm - 1.) && v1 <=my_mesh->xx[my_mesh->Mx-1]) {
				sum += S0(v1, v) * v1*v1 * f_interp(v1, theta) / xi2;
			}
		}
	}

	sum *= 2.*(dtheta0)/(2.*PI);  // cos(theta) is symmetric about 0, so double the [0, theta0] range to cover [-theta0, theta0]

	return 0.5*beta * sum * fabs(xi)/v;  // already multiplied by v
};


/* compute the avalanche term */
void knockon_chiu_ba::update(Field **xx, AppCtx *user)
{
	int i, j;
	PetscReal vip12, xij;

	PetscPrintf(PETSC_COMM_WORLD,"Update Chiu's avalanche source\n");

	double hy;
	double theta, xi, xic0, xit;
	xic0 = ba_ctx->xic*ba_ctx->xic;
	double eps = xic0/(2.-xic0), b;

	/* updating the F(p, theta) function */
	std::fill(fv.begin(), fv.end(), 0.);

	for(int idx=0; idx<ntheta; idx++)
	{
		theta = (idx)*dtheta; // from 0 to PI

		xic0 = sqrt(eps*(1.-cos(theta))/(1.-eps*cos(theta))); // the local pitch-angle limit (at theta=0) to reach theta angle
		b = (1.-eps*cos(theta))/(1.-eps);

		for(i=0; i<my_mesh->xm; i++) { //v
			temp[i] = 0;
			temp1[i] = 0;
		}

		for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) {

			if (j >= my_mesh->My2)
				xij = my_mesh->yf[j]; // cell lower face
			else
				xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);

			if (xij < -xic0-1.e-4) { // if the electron can get to the theta location

				// limited cases; j < my_mesh->My2 is already satisfied
				if (j==0)
					hy = my_mesh->yf[0] + 1.0;
				else
					hy = my_mesh->yf[j] - my_mesh->yf[j-1];

				if (my_mesh->yf[j] < -xic0) { // if the whole cell is away from the singular point
					xi = sqrt( 1. - b * (1. - my_mesh->yy[j]*my_mesh->yy[j]) ); // compute the local pitch-angle at theta

					for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) {
						vip12 = my_mesh->xx[i]; //(PetscScalar)(my_mesh->xmin + (i+0.5)*hx);
						temp[i-my_mesh->xs] += xx[j][i].fn[0]*fabs(my_mesh->yy[j])/xi * b * vip12*hy;     // xx[j][i] is in fact f * v * ra here, and see definition of F(p) in Chiu's paper and my note
					}

				} else { // if the cell is intercepted by -xic0
					xit = 0.5*( xij + (-xic0) );
					xi = sqrt( 1. - b * (1. - xit*xit) ); // compute the local pitch-angle at theta
					hy = (-xic0) - xij;

					for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) {
						vip12 = my_mesh->xx[i];
						temp[i-my_mesh->xs] += xx[j][i].fn[0]*fabs(xit)/xi * b * vip12*hy;  // xx[j][i] is in fact f * v * ra here, and see definition of F(p) in Chiu's paper and my note
					}
				}
			}// end if xi0 can reach theta

		}   // end for theta

		int size;
		MPI_Comm_size(commy, &size);
		if (size>1) {
			MPI_Allreduce(temp.get(), temp1.get(), my_mesh->xm, MPI_DOUBLE, MPI_SUM, commy); // obtain the total RE density
			MPI_Barrier(PETSC_COMM_WORLD);

			MPI_Allgatherv(temp1.get(), my_mesh->xm, MPI_DOUBLE, fv.data()+idx*my_mesh->Mx, recv_count.get(), displs.get(), MPI_DOUBLE, commx);
			MPI_Barrier(PETSC_COMM_WORLD);
		} else {
			MPI_Allgatherv(temp.get(), my_mesh->xm, MPI_DOUBLE, fv.data()+idx*my_mesh->Mx, recv_count.get(), displs.get(), MPI_DOUBLE, commx);
			MPI_Barrier(PETSC_COMM_WORLD);
		}
	}

	/* update source term on the mesh */
	double kappa, xil, xih, delta;
	double xijp1, vi, vip1, v;

	int nsamples(100);

	std::fill(src.begin(), src.end(), 0.);
	/* use quadrature rule to compute source on the nodes */

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		if (j >= my_mesh->My2)
			xij = my_mesh->yf[j]; // cell lower face
		else
			xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);

		if (j < my_mesh->My2)
			xijp1 = my_mesh->yf[j];   // cell upper face
		else
			xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j+1]) : (1.0);

	for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) {

		vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
//		vip12 = my_mesh->xx[i];
		vip1 =  my_mesh->xf[i];

		xit = -vi/(sqrt(vi*vi + 1.) + 1.);

		if (vi > pc && xij<xit) {
            delta = 1.0;
            xih = xijp1;
            if (xijp1>xit) {
               delta = (xit-xij)/(xijp1-xij);
               xih = xit;
            }
            for (int l=0; l<nsamples; l++) {

                    v = dist(gen)*(vip1-vi) + vi;
                    xi = dist(gen)*(xih-xij) + xij;

                    xit = -v/(sqrt(v*v + 1.)+1.);
                    if (xi<xit) { // reject illegal samples
                            src[(i-my_mesh->xs)+j*my_mesh->xm] += Eval(v, xi) * delta/nsamples;
                    }
            }
		}

//		for (int l = 0; l<3; l++) {  // quadrature on v
//
//			double v = 0.5*(vip1+vi) + 0.5*av[l]*(vip1-vi);
//
//			xit = -v/(sqrt(v*v + 1.) + 1.);  // source stays below the R-P curve
//			int jt = find_j(xit);
//
//			for (int j=0; j<=jt+1; j++) { // only account for co-propagating secondary electrons
//
//				if (j==0) {
//					xil = -1.; xih = my_mesh->yf[0];
//				} else {
//					xil = my_mesh->yf[j-1]; xih = my_mesh->yf[j];
//				}
//
//				delta = 1.0;
//				if (xih > xit) { // cell intercepted by the R-P curve
//					delta = (xit-xil)/(xih-xil);
//					xih = xit; // guaranteed below xi=0
//				}
//
//				if (xih > xil) {
//
//					for (int k=0; k<nsamples; ++k) {
//						xi = dist(gen)*(xih-xil) + xil;
//						src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.5*wv[l]*Eval(v, xi) * delta/nsamples;
//					}

//					xi = 0.5*(xil+xih) + 0.5*a[0]*(xih-xil);
//					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[0]*Eval(v, xi) * delta * wv[l];
//
//					xi = 0.5*(xil+xih) + 0.5*a[1]*(xih-xil);
//					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[1]*Eval(v, xi) * delta * wv[l];
//
//					xi = 0.5*(xil+xih) + 0.5*a[2]*(xih-xil);
//					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[2]*Eval(v, xi) * delta * wv[l];
//
//					xi = 0.5*(xil+xih) + 0.5*a[3]*(xih-xil);
//					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[3]*Eval(v, xi) * delta * wv[l];
//
//					xi = 0.5*(xil+xih) + 0.5*a[4]*(xih-xil);
//					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[4]*Eval(v, xi) * delta * wv[l];

//					} else {  // trapped
//						xi = 0.5*(xil+xih) - 0.5*a[0]*(xih-xil);
//						kappa = 1. + (xi*xi/(ba_ctx->xic*ba_ctx->xic) - 1.)/(1. - xi*xi);
//						theta0 = 2.*asin(kappa); // largest theta the trapped electron can reach
//						src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.5*w[0] * Eval(theta0, vip12, xi) * delta;
//
//						xi = 0.5*(xil+xih) - 0.5*a[1]*(xih-xil);
//						kappa = 1. + (xi*xi/(ba_ctx->xic*ba_ctx->xic) - 1.)/(1. - xi*xi);
//						theta0 = 2.*asin(kappa); // largest theta the trapped electron can reach
//						src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.5*w[1] * Eval(theta0, vip12, xi) * delta;
//
//						xi = 0.5*(xil+xih) + 0.5*a[1]*(xih-xil);
//						kappa = 1. + (xi*xi/(ba_ctx->xic*ba_ctx->xic) - 1.)/(1. - xi*xi);
//						theta0 = 2.*asin(kappa); // largest theta the trapped electron can reach
//						src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.5*w[1] * Eval(theta0, vip12, xi) * delta;
//
//						xi = 0.5*(xil+xih) + 0.5*a[0]*(xih-xil);
//						kappa = 1. + (xi*xi/(ba_ctx->xic*ba_ctx->xic) - 1.)/(1. - xi*xi);
//						theta0 = 2.*asin(kappa); // largest theta the trapped electron can reach
//						src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.5*w[0] * Eval(theta0, vip12, xi) * delta;
//
//					}
//				}
//			}
//		}
		}
	}

	//	for (j=0; j<my_mesh->My; j++) { //xi
	//		xijp12 = my_mesh->yy[j];
	//
	//		if (j < my_mesh->My2) { // only account for co-propagating secondary electrons
	//
	//			if (j < my_mesh->My1) { //passing
	//				for(i=0; i<my_mesh->Mx; i++) { //v
	//					vip12 = my_mesh->xx[i];
	//
	//					if (vip12 > pc)
	//						src[i+j*my_mesh->Mx] = 0.5*beta * baverage(PI, vip12, xijp12) * (-xijp12)/vip12;   // ALREADY MULTIPLIED BY V
	//				}
	//			} else {  // trapped
	//				kappa = 1. + (xijp12*xijp12/(ba_ctx->xic*ba_ctx->xic) - 1.)/(1. - xijp12*xijp12);
	//				theta0 = 2.*asin(kappa); // largest theta the trapped electron can reach
	//
	//				for(i=0; i<my_mesh->Mx; i++) { //v
	//					vip12 = my_mesh->xx[i];
	//
	//					// todo: consider if a factor of 2 makes difference (it seems there should be no factor 2)
	//					if (vip12 >pc)
	//						src[i+j*my_mesh->Mx] = 0.5*beta * baverage(theta0, vip12, xijp12) * (-xijp12)/vip12;// ALREADY MULTIPLIED BY V
	//				}
	//			}
	//		}
	//	}

};

// 1d linear interpolation
//PetscReal knockon_chiu_ba::f_interp(PetscReal v, int k)
//{
//	int ju, jm, jl;
//	jl = 0;
//	ju = my_mesh->Mx-1;
//	while ( (ju-jl) > 1) {
//		jm = (ju+jl) >> 1;
//		if (v >= my_mesh->xx[jm]/*(my_mesh->xmin + (jm+0.5)*my_mesh->hx)*/)
//			jl = jm;
//		else
//			ju = jm;
//	}
//
//	return fv[jl + k*my_mesh->Mx] + (v - my_mesh->xx[jl])/(my_mesh->xx[jl+1]-my_mesh->xx[jl]) * (fv[jl+1 + k*my_mesh->Mx] - fv[jl + k*my_mesh->Mx]);
//};

/* 2d linear interpolator */
PetscReal knockon_chiu_ba::f_interp(PetscReal v, PetscReal theta_)
{
	int ju, jm, jl;
	jl = 0;	ju = my_mesh->Mx-1;
	while ( (ju-jl) > 1) {
		jm = (ju+jl) >> 1;
		if (v >= my_mesh->xx[jm]/*(my_mesh->xmin + (jm+0.5)*my_mesh->hx)*/)
			jl = jm;
		else
			ju = jm;
	}

	int k = floor(theta_/dtheta);

	double f1 = fv[jl + k*my_mesh->Mx] + (v - my_mesh->xx[jl])/(my_mesh->xx[jl+1]-my_mesh->xx[jl]) * (fv[jl+1 + k*my_mesh->Mx] - fv[jl + k*my_mesh->Mx]);
	double f2 = fv[jl + (k+1)*my_mesh->Mx] + (v - my_mesh->xx[jl])/(my_mesh->xx[jl+1]-my_mesh->xx[jl]) * (fv[jl+1 + (k+1)*my_mesh->Mx] - fv[jl + (k+1)*my_mesh->Mx]);
	return ( ((k+1)*dtheta - theta_)*f1 + (theta_ - k*dtheta)*f2 )/dtheta;
};

PetscScalar knockon_chiu_ba::S0(PetscScalar ve, PetscScalar v)
{
	PetscScalar tmp;
	PetscScalar gm = sqrt(v*v + 1.0), gme = sqrt(ve*ve + 1.0);
	PetscScalar x = (gme-1)*(gme-1.)/(gm-1.)/(gme-gm);  // rely on external check for : gme > gm
	tmp = gme/(gme-1.);

	return v/gm * tmp*tmp / (gme*gme-1.) * (x*x - 3.*x + (1+x)/tmp/tmp);
};


// perform poloidal integration of the source
//float knockon_chiu_ba::baverage(int nsub, float v, float xi)
//{
//	float theta;
//	PetscReal gm=sqrt(v*v+1.), xi2, gm1, v1;
//
//	float sum = 0.;
//	for(int idx=0; idx<nsub; idx++) // loop through theta range
//	{
//		theta = (idx+0.5)*dtheta;
//
//		/* computing the primary electron energy */
//		xi2 = 1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xi*xi); // secondary electron pitch-angle at theta
//		gm1 = 1. + 2./(xi2*(gm+1.)/(gm-1.) - 1.); // primary energy at theta
//		v1 = sqrt(gm1*gm1 - 1.);
//
//		if (gm1 > (2.*gm - 1.) && v1 <=my_mesh->xx[my_mesh->Mx-1]) {
//			sum += S0(v1, v) * v1*v1 * f_interp(v1, idx) / xi2;
//		}
//	}
//
//	sum *= 2.*(dtheta)/(2.*PI);  // cos(theta) is symmetric about 0, so double the [0, theta0] range to cover [-theta0, theta0]
//
//	return sum;
//};

/* perform poloidal integration of the source: v, xi momentum given at theta=0*/
float knockon_chiu_ba::baverage(float theta0, float v, float xi)
{
	PetscReal gm=sqrt(v*v+1.), xi2, gm1, v1;

	float sum = 0.;
	int ntheta = 100;
	float dtheta0 = theta0/(ntheta-1), theta;

	for(int idx=0; idx<(ntheta-1); idx++) // loop through theta range
	{
		theta = (idx+0.5)*dtheta0;

		/* computing the primary electron energy */
		xi2 = 1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xi*xi); // secondary electron pitch-angle at theta
		gm1 = 1. + 2./(xi2*(gm+1.)/(gm-1.) - 1.); // primary energy at theta

		if (gm1 > 1.) { // below rp curve
			v1 = sqrt(gm1*gm1 - 1.);

			if (gm1 > (2.*gm - 1.) && v1 <=my_mesh->xx[my_mesh->Mx-1]) {
				sum += S0(v1, v) * v1*v1 * f_interp(v1, theta) / xi2;
			}
		}
	}

	sum *= 2.*(dtheta0)/(2.*PI);  // cos(theta) is symmetric about 0, so double the [0, theta0] range to cover [-theta0, theta0]

	return 0.5*beta * sum * (-xi)/v;
};

