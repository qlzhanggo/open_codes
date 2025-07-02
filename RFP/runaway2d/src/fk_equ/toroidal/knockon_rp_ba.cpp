/*
 * knockon_rp_ba.cpp
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */
#include "knockon_rp_ba.h"
#include <random>

struct xy {
	double x, y;
};

knockon_rp_ba::knockon_rp_ba(mesh *mesh_, BACtx *ba_ctx_, double beta_, double pc_):
						my_mesh(mesh_), ba_ctx(ba_ctx_), nr(0.), beta(beta_), pc(pc_)
{
	PetscPrintf(PETSC_COMM_WORLD,"Using bounce-averaged rosenbluth-putvinski knock-on source:\n");
	PetscPrintf(PETSC_COMM_WORLD,"beta = %lg\n", (beta));
	PetscPrintf(PETSC_COMM_WORLD,"pc = %lg\n", (pc));

	// ju.resize(my_mesh->xm);
	// jl.resize(my_mesh->xm);

	// PetscReal gm, xi1, xi2;
	// int ju0, jm, jl0;
	// for(int i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
	//   // velocities evaluated at j, at j+1 and j+1/2.
	//   vip12 = my_mesh->xx[i]; //(PetscScalar)(my_mesh->xmin + (i+0.5)*hx);
	//   gm = sqrt(vip12*vip12 + 1.);
	//   xi1 = -vip12/(gm+1.); //-sqrt((gm-1.)/(gm+1.));
	//   xi2 = -sqrt(xi1*xi1*(1.-xic*xic) + xic*xic);//-sqrt(xi1*xi1*(1.-eps_)/(1.+eps_) + xic*xic);

	//   // find the pitch-angle interval to set pitch-angle distribution of the source
	//   jl0 = 0;
	//   ju0 = my_mesh->My-1;
	//   while ( (ju0-jl0) > 1) {
	//     jm = (ju0+jl0) >> 1;
	//     if (xi1 >= my_mesh->yy[jm])
	// 	jl0 = jm;
	//     else
	// 	ju0 = jm;
	//   }
	//   ju[i-my_mesh->xs] = jl0; // the upper limit of the band

	//   jl0 = 0;
	//   ju0 = my_mesh->My-1;
	//   while ( (ju0-jl0) > 1) {
	//     jm = (ju0+jl0) >> 1;
	//     if (xi2 >= my_mesh->yy[jm])
	// 	jl0 = jm;
	//     else
	// 	ju0 = jm;
	//   }
	//   jl[i-my_mesh->xs] = ju0; // upper limit of the band
	// }

	// for n=4 Gauss-quadrature rule
	a[0] = -sqrt(5. + 2.*sqrt(10./7.))/3.0; a[1] = -sqrt(5. - 2.*sqrt(10./7.))/3.0; a[2] = 0.; a[3] = -a[1]; a[4] = -a[0];
	w[0] = (322. - 13.*sqrt(70))/900.; w[1] = (322. - 13.*sqrt(70))/900.; w[2] = 128./225.; w[3] = w[1]; w[4] = w[0];

	// n=3
	av[0] = -sqrt(0.6); av[1] = 0; av[2] = -av[0];
	wv[0] = 5./9.; wv[1] = 8.0/9.0; wv[2] = wv[0];

	src.resize(my_mesh->xm * my_mesh->My2, 0.);
    // = std::unique_ptr<double[]>(new double[my_mesh->xm * (my_mesh->My2-1)]);
	Eval();


}


/*
 * Evaluate the rosenbluth source
 * @v @xi the momentum space variables; @i &j the index in momentum space grid, redudent
 */
void knockon_rp_ba::Eval()
{
	PetscReal gm, xi1, xi2;
	double xij, xijp12, xijp1, vi, vip12, vip1;

	double delta, hy, xil, xih, s, xi, xit;
	double xic = ba_ctx->xic;

	double av[3], wv[3];
	av[0] = -sqrt(0.6); av[1] = 0; av[2] = -av[0];
	wv[0] = 5./9.; wv[1] = 8.0/9.0; wv[2] = wv[0];

	std::fill(src.begin(), src.end(), 0.);
	/* use quadrature rule to compute source on the nodes */
	for(int i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) {

		vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
		vip12 = my_mesh->xx[i];
		vip1 =  my_mesh->xf[i];

		if (vip12 > pc) {

		for (int l = 0; l<3; l++) {  // quadrature on v

			double v = 0.5*(vip1+vi) + 0.5*av[l]*(vip1-vi);

			gm = sqrt(vip12*vip12 + 1.);
			xi1 = -vip12/(gm+1.); //-sqrt((gm-1.)/(gm+1.));
			xi2 = -sqrt(xi1*xi1*(1.-xic*xic) + xic*xic);

			// find the pitch-angle interval to set pitch-angle distribution of the source
			int jl1 = find_j(xi1), jl2 = find_j(xi2);

			for (int j=jl1; j<=jl2+1; j++) { // only account for co-propagating secondary electrons

				if (j==0) {
					xil = -1.; xih = my_mesh->yf[0];
				} else {
					xil = my_mesh->yf[j-1]; xih = my_mesh->yf[j];
				}

				delta = 1.0;
				if (xih > xit) { // cell intercepted by the R-P curve
					delta = (xit-xil)/(xih-xil);
					xih = xit; // guaranteed below xi=0
				}

				if (xih > xil) {

					xi = 0.5*(xil+xih) + 0.5*a[0]*(xih-xil);
					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[0]* delta * wv[l];
					xi = 0.5*(xil+xih) + 0.5*a[1]*(xih-xil);
					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[1] * delta * wv[l];
					xi = 0.5*(xil+xih) + 0.5*a[2]*(xih-xil);
					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[2] * delta * wv[l];
					xi = 0.5*(xil+xih) + 0.5*a[3]*(xih-xil);
					src[(i-my_mesh->xs)+j*my_mesh->xm] += 0.25 * w[3] * delta * wv[l];

				}
			}
		}
		}
	}

//	std::vector<xy> mc;
//	int nsamples(80000);
//
//	mc.resize(nsamples);
//
//	for(int i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
//		vip12 = my_mesh->xx[i];
//
//		gm = sqrt(vip12*vip12 + 1.);
//		xi1 = -vip12/(gm+1.); //-sqrt((gm-1.)/(gm+1.));
//		xi2 = -sqrt(xi1*xi1*(1.-xic*xic) + xic*xic);
//
//	    std::random_device rd;  //Will be used to obtain a seed for the random number engine
//	    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
//	    std::uniform_real_distribution<double> dist(0.0001, 1.0);
//
//		for (int l=0; l<nsamples; ++l) {
//			mc[l].x = dist(gen)*(xi1-xi2) + xi2;
//			delta = 2.*(-mc[l].x)/PI/sqrt((mc[l].x*mc[l].x - xi1*xi1)*(xi2*xi2 - mc[l].x*mc[l].x)) * sqrt(1.-xic*xic);
//			mc[l].y = 0.5*beta / (gm*(gm-1)*(gm-1)) * delta * (xi1-xi2)/nsamples;
//		}
//
//		// now assign the monte carlo samples to grid points with proper weight
//		for (int l=0; l<nsamples; ++l) {
//
//			int j = find_j(mc[l].x);
//			if (j >= my_mesh->My2)
//				PetscPrintf(PETSC_COMM_WORLD,"This is nuts, how can xi become positive. \n");
//
//			if (j == my_mesh->My2-1) {  // near xi = 0
//				hy = 0.5*(my_mesh->yy[j] - my_mesh->yy[j-1]) - my_mesh->yy[j];
//				src[(i-my_mesh->xs)*(my_mesh->My2) + j] += mc[l].y / hy;
//
////			} else if (j == my_mesh->My1-1) { // between trap-passing
////				if (mc[l].x >= -xic) { // trapped region
////					hy = 0.5*(my_mesh->yy[j+2] - my_mesh->yy[j]);
////					source[(i-my_mesh->xs)*(my_mesh->My2) + j+1] += mc[l].y / hy;
////				} else { // passing region
////					hy = 0.5*(my_mesh->yy[j+1] - my_mesh->yy[j-1]);
////					source[(i-my_mesh->xs)*(my_mesh->My2) + j] += mc[l].y / hy;
////				}
//
//			} else {
//				delta = fabs(mc[l].x - my_mesh->yy[j]) / (my_mesh->yy[j+1] - my_mesh->yy[j]);
//
//				// lower grid point
//				if (j==0)
//					hy = 0.5*(my_mesh->yy[1] - my_mesh->yy[0]) + my_mesh->yy[j] + 1.0;
//				else
//					hy = 0.5*(my_mesh->yy[j+1] - my_mesh->yy[j-1]);
//				source[(i-my_mesh->xs)*(my_mesh->My2) + j] += mc[l].y * (1. - delta)/hy;
//
//				// upper grid point
//				if (j==my_mesh->My2-2)
//					hy = 0.5*(my_mesh->yy[j+1] - my_mesh->yy[j]) - my_mesh->yy[j+1];
//				else
//					hy = 0.5*(my_mesh->yy[j+2] - my_mesh->yy[j]);
//				src[(i-my_mesh->xs)*(my_mesh->My2) + j + 1] += mc[l].y * (delta)/hy;
//			}
//		}
//	}


	//	// find the pitch-angle interval to set pitch-angle distribution of the source
	//	int jl1 = find_j(xi1);
	//	int jl2 = find_j(xi2);
	//
	//	if (jl1 < jl2+2) {   // the bounce-averaged source spreads in xi is too narrow
	//		jl1 = (jl1>jl2) ? (jl1+1) : (jl2+1);
	//		if (jl1 >= my_mesh->My2)
	//			xih = 0;
	//
	//		if (xi>= my_mesh->yy[jl2] && xi <= xih) { // broaden the narrow distribution
	//			hy = xih - my_mesh->yy[jl2];
	//			delta = (1. - fabs(xi - 0.5*(xi1+xi2))/hy) / hy; // approximating the delta function
	//			// it should be devided by 4\pi, but nr can absorb it for simplicity
	//			return beta/(2.0) * nr / (gm*(gm-1)*(gm-1)) * delta * sqrt(1.-xic*xic); // the equation has been multiplied by v
	//		} else
	//			return 0.;
	//
	//	} else {
	//
	//		//bounce averaged source term is in a band of pitch-angle
	//		if (xi>xi2 && xi<xi1) {
	//			delta = 2.*(-xi)/PI/sqrt((xi*xi - xi1*xi1)*(xi2*xi2-xi*xi)) * sqrt(1.-xic*xic); //(1. - fabs(xi1 - my_mesh->yy[j])/(my_mesh->yy[ju] - my_mesh->yy[jl])) / hy; // the pitch-angle band distribution
	//
	//			// it should be devided by 4\pi, but nr can absorb the 2\pi for simplicity
	//			return 0.5*beta * nr / (gm*(gm-1)*(gm-1)) * delta; // the equation has been multiplied by v
	//		} else
	//			return 0.;
	//	}
};


/* compute the avalanche term from the local phase-space */
void knockon_rp_ba::update(Field **xx, AppCtx *user)
{
	int i, j;
	double hx, hy, xijp12, vip12, xij, xijp1, vi, vip1;

	PetscReal nr_buf(0.);

	PetscPrintf(PETSC_COMM_WORLD,"Update Rosenbluth-Putvinskii avalanche source\n");

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
			vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
			vip12 = my_mesh->xx[i];
			vip1 =  my_mesh->xf[i];

			if (vip12 > pc && xijp12<0) // only account for passing particles
			{
//				if (my_mesh->yy[j+1] < -ba_ctx->xic) {
//					nr_buf += xx[j][i].fn[0]*vip12*hx*hy;  // xx[j][i] is in fact f * v * ra here
//				} else {
//					hy = 0.5*(my_mesh->yy[j] - my_mesh->yy[j-1]) + (-ba_ctx->xic - my_mesh->yy[j]);

					nr_buf += xx[j][i].fn[0]*vip12*hx*hy;
//				}
			}
		}
	}
	//	nr_buf *= 2*PI;  //in the jacobian , as the Maxwellian is normalized. we absorb it here for simplicity

	MPI_Allreduce(&nr_buf, &nr, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD); // obtain the total RE density
	MPI_Barrier(PETSC_COMM_WORLD);

};
