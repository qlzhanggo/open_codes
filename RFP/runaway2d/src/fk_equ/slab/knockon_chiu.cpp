/*
 * knockon_chiu.cpp
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */
#include "knockon_chiu.h"

knockon_chiu::knockon_chiu(mesh *mesh_, double beta_, double pc_) : my_mesh(mesh_), beta(beta_), pc(pc_), dist(0., 1.)
{
	PetscPrintf(PETSC_COMM_WORLD,"Using slab Chiu knock-on source:\n");
	PetscPrintf(PETSC_COMM_WORLD,"beta = %lg\n", (beta));
	PetscPrintf(PETSC_COMM_WORLD,"pc = %lg\n", (pc));

	fv.resize(my_mesh->Mx, 0.);
	src.resize(my_mesh->xm * my_mesh->My, 0.);

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
	// MPI_Allgather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT, commx);
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
	//a[0] = -sqrt(5. + 2.*sqrt(10./7.))/3.0; a[1] = -sqrt(5. - 2.*sqrt(10./7.))/3.0; a[2] = 0.; a[3] = -a[1]; a[4] = -a[0];
	//w[0] = (322. - 13.*sqrt(70))/900.; w[1] = (322. + 13.*sqrt(70))/900.; w[2] = 128./225.; w[3] = w[1]; w[4] = w[0];

	// n=3
	//av[0] = -sqrt(0.6); av[1] = 0; av[2] = -av[0];
	//wv[0] = 5./9.; wv[1] = 8.0/9.0; wv[2] = wv[0];

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	gen = std::mt19937(rd());
}


PetscReal knockon_chiu::Eval(const PetscReal &v, const PetscReal &xi)
{
	PetscReal gm, gm1, v1;

	gm = sqrt(v*v + 1.);
	gm1 = 1.0 + 2.0/((gm+1.0)/(gm-1.0)*xi*xi - 1.);

	if (gm1 > 1.0) { // chiu source is below the rp source curve
		v1 = sqrt(gm1*gm1 - 1.0);
		if ((v1 <= my_mesh->xx[my_mesh->Mx-1]) && (gm1 > (2.*gm-1.)) && (xi < 0.)) {
			// devided by 2.0 due to the 4\pi\epsilon_0 and 2\pi r_e^2
            // (xi < 0.) is likely not needed
			return  0.5*beta * f_interp(v1) * v1*v1/v /fabs(xi) * S0(v1, v); // the equation has been multiplied by v
		} else
			return 0.;

	} else
		return 0.;
}


/* compute the avalanche term from the local phase-space */
void knockon_chiu::update(Field **xx, AppCtx *user)
{
	int i, j;
	PetscReal vip12, xijp12;

	PetscPrintf(PETSC_COMM_WORLD,"Compute Avalanche source\n");

	double hy;

	for(i=0; i<my_mesh->xm; i++) { //v
		temp[i] = 0;
	}

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		if (my_mesh->yy[j] <= 0.) {  // limit to xi<0 range
			if (j==0)
				hy = my_mesh->yf[0] + 1.0;
			else
				hy = my_mesh->yf[j] - my_mesh->yf[j-1];

			for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
				vip12 = my_mesh->xx[i];
				temp[i-my_mesh->xs] += xx[j][i].fn[0]*vip12*hy;  // xx[j][i] is in fact f * v * ra here and see definition of F(p) in Chiu's paper and my note
			}
		}
	}

	int size;
	MPI_Comm_size(commy, &size);
	if (size>1) {
		MPI_Allreduce(temp.get(), temp1.get(), my_mesh->xm, MPI_DOUBLE, MPI_SUM, commy); // obtain the total RE density
		MPI_Barrier(PETSC_COMM_WORLD);

		MPI_Allgatherv(temp1.get(), my_mesh->xm, MPI_DOUBLE, fv.data(), recv_count.get(), displs.get(), MPI_DOUBLE, commx);
		MPI_Barrier(PETSC_COMM_WORLD);
	} else {
		MPI_Allgatherv(temp.get(), my_mesh->xm, MPI_DOUBLE, fv.data(), recv_count.get(), displs.get(), MPI_DOUBLE, commx);
		MPI_Barrier(PETSC_COMM_WORLD);
	}

	int nsamples(100);

	/* now update the source term on the mesh*/
	double xil, xih, xit, delta, v, xi;
	double xij, xijp1, vi, vip1;

	std::fill(src.begin(), src.end(), 0.);

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		xij = (j>0) ? (my_mesh->yf[j-1]) : (-1.0);
		xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j]) : (1.0);

	    for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
	    	vi = (i==0) ? (my_mesh->xmin) : (my_mesh->xf[i-1]);
//	    	vip12 = my_mesh->xx[i];
	    	vip1 =  my_mesh->xf[i];

	    	xit = -vi/(sqrt(vi*vi + 1.) + 1.);

	    	if (vi > pc && xit > xij) {  // secondary electron lower limit

                delta = 1.0;
                xih = xijp1;
                if (xijp1>xit) {
                   delta = (xit-xij)/(xijp1-xij);
                   xih = xit;
                }

                //this is using Monte Carlo to compute the integral
	    		for (int l=0; l<nsamples; l++) {
	    			v = dist(gen)*(vip1-vi) + vi;
	    			xi = dist(gen)*(xih-xij) + xij;

	    			xit = -v/(sqrt(v*v + 1.)+1.);
	    			if (xi<xit) {
	    				src[(i-my_mesh->xs)+j*my_mesh->xm] += Eval(v, xi) * delta/nsamples;
	    			}
	    	    }

                //if (src[(i-my_mesh->xs)+j*my_mesh->xm]<0.0) PetscPrintf(PETSC_COMM_SELF,"%g ", src[(i-my_mesh->xs)+j*my_mesh->xm]);

	    	}
	    }
	}
}

// 1d linear interpolation
PetscReal knockon_chiu::f_interp(PetscReal v)
{
	int ju, jm, jl;
	jl = 0;
	ju = my_mesh->Mx-1;
	while ( (ju-jl) > 1) {
		jm = (ju+jl) >> 1;
		if (v >= my_mesh->xx[jm]/*(my_mesh->xmin + (jm+0.5)*my_mesh->hx)*/)
			jl = jm;
		else
			ju = jm;
	}

	return fv[jl] + (v - my_mesh->xx[jl])/(my_mesh->xx[jl+1] - my_mesh->xx[jl]) * (fv[jl+1] - fv[jl]);
}

