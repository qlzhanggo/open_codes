/*
 * knockon_rp_ba.h
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */

#ifndef KNOCKON_RP_BA_H_
#define KNOCKON_RP_BA_H_

#include "../userdata.h"
#include "../../mesh.h"
#include <memory>
#include <random>

class knockon_rp_ba
{
private:
	mesh* my_mesh;
	BACtx* ba_ctx;
	PetscReal nr;
	PetscReal beta, pc;

	std::vector<double> src;

	double a[5], w[5]; //for n=5 Gauss quadrature integral
	double av[3], wv[3];  // for n=3 Gauss quadrature integral

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

	//  std::vector<double> ju, jl;
	void Eval();

public:
	knockon_rp_ba(mesh *mesh_, BACtx *ba_ctx_, double beta_, double pc_);

	PetscReal get_src(const int i, const int j)
	{
		if (j< my_mesh->My2)
			return src[(i-my_mesh->xs) + j*my_mesh->xm];
		else
			return 0.;

//		if (j < my_mesh->My2 && j >= my_mesh->My1 )
//			return source[(i-my_mesh->xs)*(my_mesh->My2) + j] * nr;
//		else if (j < my_mesh->My1)
//			return source[(i-my_mesh->xs)*(my_mesh->My2) + j] * nr;
//		else
//			return 0.;
	}

	void update(Field **xx, AppCtx *user);

	~knockon_rp_ba() { }
};


#endif /* KNOCKON_RP_BA_H_ */
