/*
 * userdata.h
 *
 *  Created on: Sep 19, 2014
 *      Author: zehuag
 */

#ifndef USER_H_
#define USER_H_

#include <petsc.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscdmshell.h>
#include <petscsys.h>

#include <assert.h>
#include <time.h>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>

#define PI acos(-1.0) //3.1415926535897932
#define MIOVERME 100.0
#define DOF 1    //number of Legendre coefficients to keep, must be greater than 1
#define TWOTHIRDS 0.666666666666667

//#define CHAND_ON 1 //use full form of Chandraskhar functions, rather than their high energy expansion

typedef struct {
  PetscScalar fn[DOF];
} Field;

class fk_equ;   //forward declaration

typedef struct {
  DM          da; // distributed array
  fk_equ      *udEqu;
  PetscScalar dt, dt0, dt1;  // time step used to cotronl the computation of avalanche term
} AppCtx;


// a class to hold the parameter and method for toroidal case with non-zero trap region
#include "../mesh.h"
struct BACtx 
{
	double xic;
	std::vector<double> zeta1_jp12, zeta4_jp12;
	std::vector<double> zeta2_jp1, zeta3_jp1;

	BACtx(mesh *mesh_): xic(sqrt(2.*mesh_->eps/(1.+mesh_->eps))), zeta1_jp12(mesh_->My), zeta2_jp1(mesh_->My), zeta3_jp1(mesh_->My), zeta4_jp12(mesh_->My)
	{
		double xijp12, xijp1;
        double eps_=mesh_->eps;

		for (int j=0; j<mesh_->My; j++) { 
			xijp12 = mesh_->yy[j];  // cell center
			if (j < mesh_->My2)
				xijp1 = mesh_->yf[j];  // cell upper face
			else
				xijp1 = (j<(mesh_->My-1)) ? (mesh_->yf[j+1]) : (1.0);

			zeta1_jp12[j] = zeta1(xijp12);
			zeta2_jp1[j] = zeta2(xijp1);

			zeta3_jp1[j] = (2.-kappa_square(xijp1))/3. * zeta2_jp1[j] + (kappa_square(xijp1)-1.)/3. * zeta1(xijp1);
			zeta4_jp12[j] = kappa_square(xijp12) * (zeta1_jp12[j] - zeta2(xijp12));
			zeta3_jp1[j] *= eps_;
			zeta4_jp12[j] *= eps_; // multiplied by eps for convenience
		}
	};

	inline double kappa_square(const double &xi)
	{
		return 1. + (xi*xi/(xic*xic) - 1.)/(1. - xi*xi);
	}

	// utility function for bounce-averaging see re-note.pdf
	inline double zeta1(const double &xi)
	{
		double kappa = sqrt( 1. + (xi*xi/(xic*xic) - 1.)/(1. - xi*xi) );

		if (kappa>1) {// passing
			return 2./PI * boost::math::ellint_1<double>(1./kappa);
		} else {
			if (kappa==1)
				return 4.*kappa/PI * boost::math::ellint_1<double>(0.9999);
			else
				return 4.*kappa/PI * boost::math::ellint_1<double>(kappa);  // trapped twice passing? -Zehua
		}

	};

	inline double zeta2(const double &xi)
	{
		double kappa = sqrt( 1. + (xi*xi/(xic*xic) - 1.)/(1. - xi*xi) );

		if (kappa>=1.) {//passing, treat as passing at trap-passing boundary for fluxes
			return 2./PI * boost::math::ellint_2<double>(1./kappa);
		} else {
			return 4./kappa/PI * (boost::math::ellint_2<double>(kappa) - (1.-kappa*kappa) * boost::math::ellint_1<double>(kappa));
		}
	};

};


#endif /* USER_H_ */
