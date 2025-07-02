/*
 * field_simple.h
 * a simple field function
 */

#ifndef SRC_FIELD_FIELD_SIMPLE_H_
#define SRC_FIELD_FIELD_SIMPLE_H_

#include "Field_EQU.h"
#include<math.h>
#include<vector>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <stdio.h>
#include <petsc.h>

class field_simple : public Field_EQU
{
public:
	field_simple() : Field_EQU(){};

	virtual void Initialize() {};

	virtual void advance(const double &t){};

	virtual void interp(const double &t){};

	virtual void dump(char *myFile) {};

    ~field_simple(){};
};
#endif /* SRC_FIELD_FIELD_SIMPLE_H_ */
