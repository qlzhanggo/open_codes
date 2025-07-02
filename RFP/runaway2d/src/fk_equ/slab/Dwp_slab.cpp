/*
 * Dwp_ba.cpp
 *
 *  Created on: Feb 13, 2018
 *      Author: zehuag
 */

#include "Dwp_slab.h"
#include <petscvec.h>
#include <fstream>


// anomolous Doppler shift n=-1 : \omega - k_\parallel v_\parallel + \Omega_e/\gamma = 0. It resonants with with RE with v_||<0, so k_|| < 0
void Dwp_slab::anomalous(Eigen::EigenSolver<Eigen::MatrixXd> &es, Eigen::MatrixXd &m, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double, 3> &D)
{
	double gm = sqrt(v*v + 1.), k1r, k, temp;

	m << 0., -(k20*k20 - wpe2*wpe2*v*v * xi*xi/gm/gm), -2.*wpe2*wpe2*v*xi/gm/gm, wpe2*wpe2/gm/gm,
			1., 0., 0., 0.,
			0., 1., 0., 0.,
			0., 0., 1., 0.;
	es.compute(m, false);
	ev = es.eigenvalues();

	for (int ii=0; ii<4; ii++) {
		k1r = std::real(ev(ii));
		if (std::imag(ev(ii)) == 0 && k1r*v*xi > 1. && k1r<0) {
			k = sqrt(k1r*k1r + k20*k20);
			temp = 2.*wpe2*w0/sqrt(2.*PI)/dk10*exp(-0.5*(k1r-k10)*(k1r-k10)/(dk10*dk10)) / fabs((k*k + k1r*k1r)/k/(wpe2) + v*xi/gm);
			//temp *= (1. + erf(-50.*(k1r-k10)/dk10/sqrt(2)));
			D[0] += temp;
			D[1] += temp * (-wpe2*v/gm/k - xi);
			D[2] += temp * (wpe2*v/gm/k + xi)*(wpe2*v/gm/k + xi);
		}
	}
}

// Cherenkov resonance n=0 : \omega - k_\parallel v_\parallel = 0
void Dwp_slab::cherenkov(Eigen::EigenSolver<Eigen::MatrixXd> &es, Eigen::MatrixXd &m, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double, 3> &D)
{
	double gm = sqrt(v*v + 1.), k1r, k, temp;

	k = v*v/(gm*gm) * xi*xi - k20*k20;
	if (k>=0) {
		k1r = -sqrt(k);
		k = fabs(v/gm * xi);
		temp = 2.*wpe2*w0/sqrt(2.*PI)/dk10*exp(-0.5*(k1r-k10)*(k1r-k10)/(dk10*dk10))/fabs((k*k + k1r*k1r)/k/(wpe2) + v*xi/gm);
		D[0] += temp;
		D[1] += temp * (-wpe2*v/gm/k - xi);
		D[2] += temp * (wpe2*v/gm/k + xi)*(wpe2*v/gm/k + xi);
	}
}

// normal Doppler shift  n=1 : \omega - k_\parallel v_\parallel - \Omega_e/\gamma = 0.
void Dwp_slab::normal(Eigen::EigenSolver<Eigen::MatrixXd> &es, Eigen::MatrixXd &m, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double, 3> &D)
{
	double gm = sqrt(v*v + 1.), k1r, k, temp;

	m << 0., -(k20*k20 - wpe2*wpe2*v*v * xi*xi/gm/gm), 2.*wpe2*wpe2*v*xi/gm/gm, wpe2*wpe2/gm/gm,
			1., 0., 0., 0.,
			0., 1., 0., 0.,
			0., 0., 1., 0.;
	es.compute(m, false);
	ev = es.eigenvalues();

	for (int ii=0; ii<4; ii++) {
		k1r = std::real(ev(ii));
		if (std::imag(ev(ii)) == 0 && k1r*v*xi > -1. && k1r>0) {
			k = sqrt(k1r*k1r + k20*k20);
			temp = 2.*wpe2*w0/sqrt(2.*PI)/dk10*exp(-0.5*(k1r-k10)*(k1r-k10)/(dk10*dk10))/fabs((k*k + k1r*k1r)/k/wpe2 - v*xi/gm);
			//temp *= (1. + erf(-50.*(k1r-k10)/dk10/sqrt(2)));
			D[0] += temp;
			D[1] += temp * (wpe2*v/gm/k - xi);
			D[2] += temp * (wpe2*v/gm/k - xi)*(wpe2*v/gm/k - xi);
		}
	}
}

void Dwp_slab::setup(double w0_, double k10_, double k20_, double dk10_, double wpe)
{
	double vip12, xijp12, gm, tmp, xi;
	int i, j, idx;

	w0=w0_; k10=k10_; k20=k20_; dk10=dk10_; wpe2 = wpe*wpe;

	std::array<double, 3> D_tmp;

	PetscPrintf(PETSC_COMM_WORLD,"Setting up the slab geometry wave-particle diffusion operator\n");

	pp.resize((my_mesh->xm +2)*(my_mesh->ym+2));
	pxi.resize((my_mesh->xm +2)*(my_mesh->ym+2));
	xixi.resize((my_mesh->xm +2)*(my_mesh->ym+2));

	Eigen::MatrixXd m(4, 4);
	Eigen::EigenSolver<Eigen::MatrixXd> es;
	Eigen::VectorXcd ev(4);

    PetscErrorCode         ierr;
	PetscViewer            outputfile;                   /* file to output data to */
    PetscScalar    pp_, pxi_, xixi_;
    PetscInt IDX;
	char filename[50];
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    //create three petsc vec for print out coefficients
    Vec            PP, PXI, XIXI;
    VecCreate(PETSC_COMM_WORLD,&PP);
    VecCreate(PETSC_COMM_WORLD,&PXI);
    VecCreate(PETSC_COMM_WORLD,&XIXI);
    VecSetSizes(PP,(my_mesh->xm)*(my_mesh->ym),PETSC_DECIDE);
    VecSetSizes(PXI,(my_mesh->xm)*(my_mesh->ym),PETSC_DECIDE);
    VecSetSizes(XIXI,(my_mesh->xm)*(my_mesh->ym),PETSC_DECIDE);
    VecSetFromOptions(PP);
    VecSetFromOptions(PXI);
    VecSetFromOptions(XIXI);



	// compute the wave-particle diffusion matrix on cell centers, values outside the domain will not be used
	for (j=my_mesh->ys-1; j<my_mesh->ys+my_mesh->ym+1; j++) { //xi

		if (j<0)
			xijp12 = -1.;
		else if (j==my_mesh->My)
			xijp12 = 1.;
		else
			xijp12 = my_mesh->yy[j];

		for (i=my_mesh->xs-1; i<my_mesh->xs+my_mesh->xm+1; i++) { //p

			idx = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2); // index of the diffusion coefficient matrix
			pp[idx] = 0;
			pxi[idx] = 0;
			xixi[idx] = 0;

			if (i<0)
				vip12 = 2.*my_mesh->xmin - my_mesh->xx[0];
			else if (i==my_mesh->Mx-1)
				vip12 = 2.*my_mesh->xmax - my_mesh->xx[my_mesh->Mx-1];
			else
				vip12 = my_mesh->xx[i];

			D_tmp.fill(0.);
			//anomalous(es, m, ev, vip12, xi, D_tmp);
			//cherenkov(es, m, ev, vip12, xi, D_tmp);
			normal(es, m, ev, vip12, xijp12, D_tmp);

			pp[idx] += D_tmp[0];
			pxi[idx] += D_tmp[1];
			xixi[idx] += D_tmp[2];

            if (i!=my_mesh->xs-1 && i!=my_mesh->xs+my_mesh->xm && j!=my_mesh->ys-1 && j!=my_mesh->ys+my_mesh->ym) {
        
               pp_=pp[idx];
               pxi_=pxi[idx];
               xixi_=xixi[idx];

               //note that i j is already a global index
			   IDX = i + j*my_mesh->Mx; // index of the diffusion coefficient matrix

               ierr=VecSetValues(PP,  1,&IDX,&pp_,ADD_VALUES); assert(ierr==0);
               ierr=VecSetValues(PXI ,1,&IDX,&pxi_,ADD_VALUES); assert(ierr==0);
               ierr=VecSetValues(XIXI,1,&IDX,&xixi_,ADD_VALUES); assert(ierr==0);
           }



		}

		}

    VecAssemblyBegin(PP);
    VecAssemblyEnd(PP);
    VecAssemblyBegin(PXI);
    VecAssemblyEnd(PXI);
    VecAssemblyBegin(XIXI);
    VecAssemblyEnd(XIXI);

	PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
	PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
	PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
	sprintf(filename, "./PP.dat");
	PetscViewerFileSetName(outputfile, filename);
	VecView(PP, outputfile);
	PetscViewerDestroy(&outputfile);

	PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
	PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
	PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
	sprintf(filename, "./PXI.dat");
	PetscViewerFileSetName(outputfile, filename);
	VecView(PXI, outputfile);
	PetscViewerDestroy(&outputfile);

	PetscViewerCreate(PETSC_COMM_WORLD, &outputfile);
	PetscViewerSetType(outputfile, PETSCVIEWERBINARY);
	PetscViewerFileSetMode(outputfile, FILE_MODE_WRITE);
	sprintf(filename, "./XIXI.dat");
	PetscViewerFileSetName(outputfile, filename);
	VecView(XIXI, outputfile);
	PetscViewerDestroy(&outputfile);


    VecDestroy(&PP);
    VecDestroy(&PXI);
    VecDestroy(&XIXI);

    if (false) {
        std::ofstream ofile;
        ofile.open("Dwp.dat");
        if (ofile.is_open()) {
        	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi
        
        		for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //p
        			idx = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2); // index of the diffusion coefficient matrix
        			//ofile << pp[idx] << " ";
                    ofile <<rank<<" "<<i<<" "<<j<<" "<<i+j*my_mesh->Mx<<" "<<pp[idx]<<" "<<pxi[idx]<<" "<<xixi[idx]<<std::endl;
	        			
        		}
        		//ofile << std::endl;
        
        	}
        }
        ofile.close();
    }

};
