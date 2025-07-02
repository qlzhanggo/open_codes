/*
 * field_pre.h
 *
 *  Created on: Oct 4, 2017
 *      Author: guo
 */

#ifndef SRC_FIELD_FIELD_PRE_H_
#define SRC_FIELD_FIELD_PRE_H_

#include "Field_EQU.h"
#include<math.h>
#include<vector>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <stdio.h>

#include <petsc.h>

struct t_E{
	double t;
	pstate state;

	t_E(const t_E& copy) : t(copy.t), state(copy.state) {};

	t_E(t_E& copy) : t(copy.t), state(copy.state) {};

	t_E() : t(0), state({0, 1, 0.1}) {};
};

/**
 * field_tp is empty to allow uniform interface for test particle simulaiton
 */
class field_pre : public Field_EQU
{
private:
    std::vector<t_E> E_table; // the pre-set state values at different time
    bool file;
public:

	field_pre() : Field_EQU(), E_table(0), file(false) {};

	field_pre(char *fname) : state({0, 1., 0.1}), E_table(0), file(true)
	{
		E_table.reserve(100);

		double temp1, temp2, temp3, temp4;
		t_E temp;
		int count(0);
		std::ifstream File;

		File.open(fname, std::ifstream::in);

		if (File.is_open()) {
			while(File >> temp1 >> temp2 >> temp3 >> temp4) {
				temp.t = temp1;
				temp.state.E = temp2;
				temp.state.Z = temp3;
				temp.state.alpha = temp4;

				E_table.push_back(temp);
				count++;
			}
			File.close();
			E_table.resize(count);

		} else {
			std::perror(("error while opening file "));
			exit(0);
		}

	};

	virtual void Initialize() {
		if (file) {
			state.E = E_table[0].state.E; state.Z = E_table[0].state.Z; state.alpha=E_table[0].state.alpha;
		}
	};

	virtual void advance(const double &t)
	{
		if (!file) {
			state.E=10.5*tanh((t-0.1)/0.02) - 9.5*tanh((t-0.5)/0.3) + 1.;
			state.Z = 1.0;
			state.alpha = 0.1;
		} else
			interp(t);
	};

	void interp(const double &t)
	{
		if (t < E_table.back().t) {
			int ju, jm, jl;
			jl = 0;
			ju = E_table.size()-1;
			while ( (ju-jl) > 1) {
				jm = (ju+jl) >> 1;
				if (t >= E_table[jm].t)
					jl = jm;
				else
					ju = jm;
			}
			double r = (t - E_table[jl].t)/(E_table[jl+1].t- E_table[jl].t);
			state.E = E_table[jl].state.E + r * (E_table[jl+1].state.E - E_table[jl].state.E);
			state.Z = E_table[jl].state.Z + r * (E_table[jl+1].state.Z - E_table[jl].state.Z);
			state.alpha = E_table[jl].state.alpha + r * (E_table[jl+1].state.alpha - E_table[jl].state.alpha);
		} else {
			state.E = E_table.back().state.E;
			state.Z = E_table.back().state.Z;
			state.alpha = E_table.back().state.alpha;
		}
	}

	virtual pstate &get_state()
	{
     	return state;
	}

	virtual void dump(char *myFile) {};

    ~field_pre(){};

};




#endif /* SRC_FIELD_FIELD_PRE_H_ */
