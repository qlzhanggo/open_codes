/*
 * knockon_chiu.h
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */

#ifndef KNOCKON_CHIU_H_
#define KNOCKON_CHIU_H_

#include "../userdata.h"
#include "../../mesh.h"


#include <memory>
#include <random>

class knockon_chiu
{
private:
	mesh* my_mesh;
	std::vector<double> fv;
	double beta, pc;

	std::vector<double> src;  // the source on the mesh

	MPI_Comm  commy, commx;

	std::mt19937 gen; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dist;

public:
	knockon_chiu(mesh *mesh_, double beta_, double pc_);

	PetscReal Eval(const PetscReal &v, const PetscReal &xi);

	PetscReal get_src(const int i, const int j)
	{
		return src[(i-my_mesh->xs) + j*my_mesh->xm];
	}

    void output(Field **xx, AppCtx *user)
    {
    	int i, j;
        for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi
    		for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { // v
    			xx[j][i].fn[0] = get_src(i, j);
    		}
    	}
    }

	void update(Field **xx, AppCtx *user);

	~knockon_chiu()
	{
		MPI_Comm_free(&commy);
		MPI_Comm_free(&commx);
	};

	std::unique_ptr<int[]> recv_count, displs;
	std::unique_ptr<double[]> temp, temp1;

protected:
	PetscReal f_interp(PetscReal v);

	int find_j(double xi)
	{
		int ju, jm, jl;
		jl = 0;
		ju = my_mesh->My-1;
		while ( (ju-jl) > 1) {
			jm = (ju+jl) >> 1;
			if (xi >= my_mesh->yy[jm])
				jl = jm;
			else
				ju = jm;
		}
		return jl;
	}

	PetscScalar S0(PetscScalar ve, PetscScalar v)
	{
		PetscScalar tmp;
		PetscScalar gm = sqrt(v*v + 1.0), gme = sqrt(ve*ve + 1.0);
		PetscScalar x = (gme-1)*(gme-1.)/(gm-1.)/(gme-gm);  // rely on external check for : gme > gm
		tmp = gme/(gme-1.);
		return v/gm * tmp*tmp / (gme*gme-1.) * (x*x - 3.*x + (1+x)/tmp/tmp);
	}

	// annihlation integral (this is called S2 in the notes and it is not used)
	PetscScalar S1(PetscScalar gm, PetscScalar gmin)
	{
		if (gm < (2.*gmin - 1.))
			return 0.;
		else
			return 1./(gm*gm-1.) * ( 0.5*(gm+1.) - gmin - gm*gm*(1./(gm-gmin) - 1./(gmin-1.)) + (2.*gm-1.)/(gm-1.) * log((gmin-1.)/(gm-gmin)) );
	}
};


#endif /* FK_CHIU_H_ */
