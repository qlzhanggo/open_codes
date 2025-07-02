/*
 * knockon_rp.cpp
 */
#include "knockon_rp.h"

knockon_rp::knockon_rp(mesh *mesh_, double beta_, double pc_) : my_mesh(mesh_), nr(0.), beta(beta_), pc(pc_)
{
	PetscPrintf(PETSC_COMM_WORLD,"Using rosenbluth-putvinski knock-on source:\n");
	PetscPrintf(PETSC_COMM_WORLD,"beta = %lg\n", (beta));
	PetscPrintf(PETSC_COMM_WORLD,"pc = %lg\n", (pc));
	source.resize(my_mesh->xm * (my_mesh->My), 0.); 
	Eval();
}

/*
 * Set the nonstiff part (collision operators + driving terms) as explicit
 * @xx the unknown variables; @ff the rhs values to be returned
 */
void knockon_rp::Eval()
{
	PetscReal gm, xi1, vip12;
	double delta, hy, s;

	for(int i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
		vip12 = my_mesh->xx[i];

		gm = sqrt(vip12*vip12 + 1.);
		xi1 = -vip12/(gm+1.); //-sqrt((gm-1.)/(gm+1.));

		// find the pitch-angle interval to set pitch-angle distribution of the source
		int ju, jm, jl;
		jl = 0;
		ju = my_mesh->My-1;
		while ( (ju-jl) > 1) {
			jm = (ju+jl) >> 1;
			if (xi1 >= my_mesh->yy[jm])
				jl = jm;
			else
				ju = jm;
		}

		if (my_mesh->yy[jl] >= 0)
			PetscPrintf(PETSC_COMM_WORLD,"This is nuts, how can xi become positive. \n");

		delta = fabs(xi1 - my_mesh->yy[jl]) / (my_mesh->yy[jl+1] - my_mesh->yy[jl]);

		// lower grid point
		if (jl==0)
			hy = 0.5*(my_mesh->yy[1] - my_mesh->yy[0]) + my_mesh->yy[jl] + 1.0;
		else
			hy = 0.5*(my_mesh->yy[jl+1] - my_mesh->yy[jl-1]);
		source[(i-my_mesh->xs)*(my_mesh->My) + jl] = 0.5*beta / (gm*(gm-1)*(gm-1)) * (1. - delta)/hy;

		// upper grid point
		if (jl==my_mesh->My-2)
			hy = 0.5*(my_mesh->yy[jl+1] - my_mesh->yy[jl]) - my_mesh->yy[jl+1];
		else
			hy = 0.5*(my_mesh->yy[jl+2] - my_mesh->yy[jl]);
		source[(i-my_mesh->xs)*(my_mesh->My) + jl + 1] = 0.5*beta / (gm*(gm-1)*(gm-1)) * (delta)/hy;
	}
	//	if (xi>= my_mesh->yy[jl] && xi <= my_mesh->yy[ju]) {
	//		hy = my_mesh->yy[ju] - my_mesh->yy[jl];
	//		delta = (1. - fabs(xi - xi1)/hy) / hy; // approximating the delta function
	//		// it should be devided by 4\pi, but nr can absorb it for simplicity
	//		return beta/(2.0) * nr / (gm*(gm-1)*(gm-1)) * delta; //(1.e-2/PI/(1.e-4 + square((xijp12 - xi1)))); // the equation has been multiplied by v
	//	} else
	//		return 0.;

	//	if (xi == my_mesh->yy[jl]) {
	//		if (jl == my_mesh->My2-1) {  // near xi = 0
	//			hy = 0.5*(my_mesh->yy[jl] - my_mesh->yy[jl-1]) - my_mesh->yy[jl];
	//			// it should be devided by 4\pi, but nr can absorb it for simplicity
	//			return 0.5*beta * nr / (gm*(gm-1)*(gm-1)) / hy;
	//		} else {
	//			if (jl==0)
	//				hy = 0.5*(my_mesh->yy[1] - my_mesh->yy[0]) + my_mesh->yy[0] + 1.0;
	//			else
	//				hy = 0.5*(my_mesh->yy[jl+1] - my_mesh->yy[jl-1]);
	//
	//			delta = (1. - fabs(xi - xi1)/(my_mesh->yy[jl+1]-my_mesh->yy[jl])) / hy; // approximating the delta function
	//			return 0.5*beta * nr / (gm*(gm-1)*(gm-1)) * delta;
	//		}
	//	} else if (xi == my_mesh->yy[jl+1]) {
	//
	//		if (jl == my_mesh->My2-2) {  // near xi = 0
	//			hy = 0.5*(my_mesh->yy[jl] - my_mesh->yy[jl-1]) - my_mesh->yy[jl];
	//		} else {
	//			hy = 0.5*(my_mesh->yy[jl+2] - my_mesh->yy[jl]);
	//		}
	//
	//		delta = (1. - fabs(xi - xi1)/(my_mesh->yy[jl+1]-my_mesh->yy[jl])) / hy; // approximating the delta function
	//		// it should be devided by 4\pi, but nr can absorb it for simplicity
	//		return 0.5*beta * nr / (gm*(gm-1)*(gm-1)) *delta;
	//	} else
	//		return 0.;

};

/* compute the avalanche term from the local phase-space */
void knockon_rp::update(Field **xx, AppCtx *user)
{
	int i, j;
	double hx, hy, vip12, xijp12;

	PetscReal nr_buf(0.);
	PetscPrintf(PETSC_COMM_WORLD,"Update Rosenbluth-Putvinskii avalanche source\n");

	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi
		xijp12 = my_mesh->yy[j];
        if (xijp12>0.0) continue;   //skip the rest
                                    
		if (j==0)
			hy = my_mesh->yf[0] + 1.0;
		else
			hy = my_mesh->yf[j] - my_mesh->yf[j-1];

		for(i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //v
			vip12 =  my_mesh->xx[i];  
			if (i==0)
				hx = my_mesh->xf[0] - my_mesh->xmin;
			else
				hx = my_mesh->xf[i] - my_mesh->xf[i-1];
			if (vip12 > pc && xijp12<=0) // only account for passing particles
				nr_buf += xx[j][i].fn[0]*vip12*hx*hy;  // xx[j][i] is in fact f * v * ra here
		}
	}

	MPI_Allreduce(&nr_buf, &nr, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD); // obtain the total RE density
	MPI_Barrier(PETSC_COMM_WORLD);
};
