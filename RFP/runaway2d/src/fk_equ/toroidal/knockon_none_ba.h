/*
 * knock_none.h
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */

#ifndef KNOCKON_NONE_BA_H_
#define KNOCKON_NONE_BA_H_

#include "../userdata.h"
#include "../../mesh.h"


class knockon_none_ba
{  
public:
	knockon_none_ba(mesh *mesh_, BACtx *ba_ctx_, double beta_, double pc_) {};

	PetscReal Eval(const PetscReal &v, const PetscReal &xi, const int i, const int j) {return 0.;};

	PetscReal get_src(const int i, const int j)
	{
		return 0.;
	}

	void update(Field **xx, AppCtx *user) {};

	~knockon_none_ba() {};

};


#endif /* KNOCKON_NONE_BA_H_ */
