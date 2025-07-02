/**
 *  @file Dwp_ba.h
 *
 *  Created on: Aug 19, 2017
 *    Author: zehuag
 */

#ifndef DWP_BA_HPP_
#define DWP_BA_HPP_

#include "../../mesh.h"
#include "../userdata.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class Dwp_ba
{
protected:
	mesh *my_mesh;
	float eps, xic;
	double w0, k10, k20, dk10, wpe2;

public:
	std::vector<double> pp, pxi, xixi;
	std::vector<double> ppFace, pxiFace, xixiFace;

	// a default contructor;
	Dwp_ba() {};

	// setting quasilinear diffusion operator
	Dwp_ba(mesh *mesh_, BACtx &ba_ctx_) : my_mesh(mesh_), xic(ba_ctx_.xic), w0(0.), k10(0.), k20(0.), dk10(0.), wpe2(0.), pp(0), pxi(0), xixi(0)
	{
		eps = xic*xic/(2.-xic*xic);
	};

	void setup(double w0, double k10, double k20, double dk10, double wpe=1.);

	~Dwp_ba(){};

private:
	void anomalous(Eigen::EigenSolver<Eigen::MatrixXd> &s, Eigen::MatrixXd &em,  Eigen::VectorXcd &ev, double &v, double &xi, std::array<double,3> &D);

	void cherenkov(Eigen::EigenSolver<Eigen::MatrixXd> &s, Eigen::MatrixXd &em, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double,3> &D);

	void normal(Eigen::EigenSolver<Eigen::MatrixXd> &s, Eigen::MatrixXd &em, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double,3> &D);
};

#endif
