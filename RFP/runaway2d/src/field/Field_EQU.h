/*
 * Field_EQU.h
 *
 *  Created on: Oct 4, 2017
 *      Author: guo
 */

#ifndef SRC_FIELD_FIELD_EQU_H_
#define SRC_FIELD_FIELD_EQU_H_


#include <vector>
#include <iostream>

struct pstate{
	double E, Z, alpha;
};

/**
 * Field_EQU is an interface for describing field equations and their solver routines
 */
class Field_EQU {
private:
    pstate state;  // the current state

public:
	Field_EQU(PetscScalar E=0.0, PetscScalar Z=1.0, PetscScalar alpha=0.1) : state({E, Z, alpha}){};

	virtual void Initialize() = 0;

	virtual void advance(const double &t) = 0;

	virtual pstate& get_state() {return state;};

	virtual void set_state(pstate state_){state=state_;};

	virtual void dump(char *myFile) = 0;

    virtual ~Field_EQU(){};
};
#endif /* SRC_FIELD_FIELD_EQU_H_ */
