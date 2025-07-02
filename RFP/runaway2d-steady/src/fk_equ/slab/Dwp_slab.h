/**
 *  @file Dwp.h
 *
 *  Created on: Aug 19, 2017
 *    Author: zehuag
 */

#ifndef DWP_SLAB_HPP_
#define DWP_SLAB_HPP_

#include "../../mesh.h"
#include "../userdata.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class Dwp_slab
{
protected:
	mesh *my_mesh;
	double w0, k10, k20, dk10, wpe2;

public:
	std::vector<double> pp, pxi, xixi;

	// a default contructor;
	Dwp_slab() {};

	// setting quasilinear diffusion operator
	Dwp_slab(mesh *mesh_) : my_mesh(mesh_), pp(0), pxi(0), xixi(0)
	{
	};

	void setup(double w0, double k10, double k20, double dk10, double wpe=1.);

        //same as setup but compute face values
	void setupFace(double w0, double k10, double k20, double dk10, double wpe=1.);

	~Dwp_slab(){};

private:
	void anomalous(Eigen::EigenSolver<Eigen::MatrixXd> &s, Eigen::MatrixXd &em,  Eigen::VectorXcd &ev, double &v, double &xi, std::array<double,3> &D);

	void cherenkov(Eigen::EigenSolver<Eigen::MatrixXd> &s, Eigen::MatrixXd &em, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double,3> &D);

	void normal(Eigen::EigenSolver<Eigen::MatrixXd> &s, Eigen::MatrixXd &em, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double,3> &D);

};

#endif

//old stuff
//	{
//		double vip12, xijp12, gm, k, k1r, temp;
//		int i, j, idx;
//		double wpe2 = wpe*wpe;
//
//		PetscPrintf(PETSC_COMM_WORLD,"Setting up the wave-particle diffusion operator\n");
//
//		pp.resize((my_mesh->xm +2)*(my_mesh->ym+2));
//		pxi.resize((my_mesh->xm +2)*(my_mesh->ym+2));
//		xixi.resize((my_mesh->xm +2)*(my_mesh->ym+2));
//
//		Eigen::MatrixXd m(4, 4);
//		Eigen::EigenSolver<Eigen::MatrixXd> es;
//		Eigen::VectorXcd ev(4);
//
//		// compute the wave-particle diffusion matrix on cell centers, values outside the domain will not be used
//		for (j=my_mesh->ys-1; j<my_mesh->ys+my_mesh->ym+1; j++) { //xi
//
//			if (j<0)
//				xijp12 = -1.;
//			else if (j==my_mesh->My)
//				xijp12 = 1.;
//			else
//				xijp12 = my_mesh->yy[j];
//
//			for (i=my_mesh->xs-1; i<my_mesh->xs+my_mesh->xm+1; i++) { //p
//
//				idx = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2);// index of the diffusion coefficient matrix
//				pp[idx] = 0;
//				pxi[idx] = 0;
//				xixi[idx] = 0;
//
//				if (i<0)
//					vip12 = 2.*my_mesh->xmin - my_mesh->xx[0];
//				else if (i==my_mesh->Mx-1)
//					vip12 = 2.*my_mesh->xmax - my_mesh->xx[my_mesh->Mx-1];
//				else
//					vip12 = my_mesh->xx[i];
//
//				gm = sqrt(vip12*vip12 + 1.) ;
//
//				// anomolous Doppler shift n=-1 to resonant with RE with v_||<0, so k_|| < 0
////				m << 0., -(k20*k20 - wpe2*wpe2*vip12*vip12 * xijp12*xijp12/gm/gm), -2.*wpe2*wpe2*vip12*xijp12/gm/gm, wpe2*wpe2/gm/gm,
////						1., 0., 0., 0.,
////						0., 1., 0., 0.,
////						0., 0., 1., 0.;
////				es.compute(m, false);
////				ev = es.eigenvalues();
////
////				for (int ii=0; ii<4; ii++) {
////					k1r = std::real(ev(ii));
////					if (std::imag(ev(ii)) == 0 && k1r*vip12*xijp12 > 1. && k1r<0) {
////						k = sqrt(k1r*k1r + k20*k20);
////						temp = 2.*wpe2*w0/sqrt(2.*PI)/dk10*exp(-0.5*(k1r-k10)*(k1r-k10)/(dk10*dk10)) / fabs((k*k + k1r*k1r)/k/(wpe2) + vip12*xijp12/gm);
////						//temp *= (1. + erf(-50.*(k1r-k10)/dk10/sqrt(2)));
////						pp[idx] += temp;
////						pxi[idx] += temp * (-wpe2*vip12/gm/k - xijp12);
////						xixi[idx] += temp * (wpe2*vip12/gm/k + xijp12)*(wpe2*vip12/gm/k + xijp12);
////					}
////				}
//
//				// Cherenkov resonance n=0
//				if (xijp12 <0) {
//					k1r = vip12*vip12/(gm*gm) * xijp12*xijp12 - k20*k20;
//					if (k1r>=0 && k10<0) {
//						k1r = -sqrt(k1r);
//						k = fabs(vip12/gm * xijp12);
//						temp = 2.*wpe2*w0/sqrt(2.*PI)/dk10*exp(-0.5*(k1r-k10)*(k1r-k10)/(dk10*dk10))/fabs((k*k + k1r*k1r)/k/(wpe2) + vip12*xijp12/gm);
//						pp[idx] += temp;
//						pxi[idx] += temp * (-wpe2*vip12/gm/k - xijp12);
//						xixi[idx] += temp * (wpe2*vip12/gm/k + xijp12)*(wpe2*vip12/gm/k + xijp12);
//					}
//				}
//
//				// normal Doppler shift
////				m << 0., -(k20*k20 - wpe2*wpe2*vip12*vip12 * xijp12*xijp12/gm/gm), 2.*wpe2*wpe2*vip12*xijp12/gm/gm, wpe2*wpe2/gm/gm,
////						1., 0., 0., 0.,
////						0., 1., 0., 0.,
////						0., 0., 1., 0.;
////				es.compute(m, false);
////				ev = es.eigenvalues();
////
////				for (int ii=0; ii<4; ii++) {
////					k1r = std::real(ev(ii));
////					if (std::imag(ev(ii)) == 0 && k1r*vip12*xijp12 > -1. && k1r>0) {
////						k = sqrt(k1r*k1r + k20*k20);
////						temp = 2.*wpe2*w0/sqrt(2.*PI)/dk10*exp(-0.5*(k1r-k10)*(k1r-k10)/(dk10*dk10))/fabs((k*k + k1r*k1r)/k/wpe2 - vip12*xijp12/gm);
////						//temp *= (1. + erf(-50.*(k1r-k10)/dk10/sqrt(2)));
////						pp[idx] += temp;
////						pxi[idx] += temp * (wpe2*vip12/gm/k - xijp12);
////						xixi[idx] += temp * (wpe2*vip12/gm/k - xijp12)*(wpe2*vip12/gm/k - xijp12);
////					}
////				}
//			}
//		}

//	};


