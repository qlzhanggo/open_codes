/*
 * knock_rp.h
 *
 *  Created on: Sep 19, 2016
 *      Author: zehuag
 */

#ifndef KNOCKON_RP_H_
#define KNOCKON_RP_H_

#include "../userdata.h"
#include "../../mesh.h"


class knockon_rp
{
private:
	mesh *my_mesh;
	double nr;
	double beta, pc;
	std::vector<double> source;
	void Eval();

public:
	knockon_rp(mesh *mesh_, double beta_, double pc_);

	void update(Field **xx, AppCtx *user);

	PetscReal get_src(const int i, const int j)
	{
		return source[(i-my_mesh->xs)*(my_mesh->My) + j] * nr;
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

	~knockon_rp()
	{
	};

};


#endif /* KNOCKON_RP_H_ */
