/*
 * knock_none.h
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */

#ifndef KNOCKON_NONE_H_
#define KNOCKON_NONE_H_

#include "../userdata.h"
#include "../../mesh.h"


class knockon_none
{  
private:
	mesh *my_mesh;

public:
	knockon_none(mesh *mesh_, double beta_, double pc_) : my_mesh(mesh_)
    {
		PetscPrintf(PETSC_COMM_WORLD,"Invoke none knockon collision\n");
    };

	PetscReal get_src(const int i, const int j)
	{
		return 0.;
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

	PetscReal Eval(const PetscReal &v, const PetscReal &xi) {return 0.;};

	void update(Field **xx, AppCtx *user) {};

	~knockon_none() {};

};


#endif /* KNOCKON_NONE_H_ */
