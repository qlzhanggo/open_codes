/*
 * knockon_chiu_ba.h
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */

#ifndef KNOCKON_CHIU_BA_H_
#define KNOCKON_CHIU_BA_H_

#include "../userdata.h"
#include "../../mesh.h"

#include <random>
#include <memory>

struct xy {
	double x, y;
};


class knockon_chiu_ba
{
private:
	mesh* my_mesh;
	BACtx *ba_ctx;
	std::vector<double> fv; // pitch-angle averaged distribution
	double beta, pc;

	std::vector<double> src;  // the source on the mesh

	MPI_Comm  commy, commx;

	float eps;
	int ntheta;
	double dtheta;

	double a[5], w[5];
	double av[3], wv[3];

	std::mt19937 gen; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dist;

public:
	knockon_chiu_ba(mesh *mesh_, BACtx *ba_ctx_, double beta_, double pc_);

	PetscReal get_src(const int i, const int j)
	{
		if (j< my_mesh->My2)
			return src[(i-my_mesh->xs) + j*my_mesh->xm];
		else
			return 0.;
	}

	void update(Field **xx, AppCtx *user);

	~knockon_chiu_ba()
	{
		MPI_Comm_free(&commy);
		MPI_Comm_free(&commx);
	};

	//    std::vector<int> recv_count, displs;
	std::unique_ptr<int[]> recv_count, displs;
	std::unique_ptr<double[]> temp, temp1;

protected:
//	PetscReal f_interp(PetscReal v, int k);
	PetscReal Eval(const double &v, const double &xi);

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

	PetscReal f_interp(PetscReal v, PetscReal theta_);

	PetscScalar S0(PetscScalar ve, PetscScalar v);

	//  // annihlation integral
	//  PetscScalar S1(PetscScalar gm, PetscScalar gmin)
	//  {
	//    if (gm < (2.*gmin - 1.))
	//      return 0.;
	//    else
	//      return 1./(gm*gm-1.) * ( 0.5*(gm+1.) - gmin - gm*gm*(1./(gm-gmin) - 1./(gmin-1.)) + (2.*gm-1.)/(gm-1.) * log((gmin-1.)/(gm-gmin)) );
	//  }

	// perform poloidal integration of the source
//	float baverage(int nsub, float v, float xi);

	// perform poloidal integration of the source
	float baverage(float theta0, float v, float xi);
};


#endif /* FK_CHIU_BA_H_ */
