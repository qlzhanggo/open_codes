/*
 * Dwp_ba.cpp
 *
 *  Created on: Feb 13, 2018
 *      Author: zehuag
 */

#include "Dwp_ba.h"
#include <fstream>

// anomolous Doppler shift n=-1 : \omega - k_\parallel v_\parallel + \Omega_e/\gamma = 0
void Dwp_ba::anomalous(Eigen::EigenSolver<Eigen::MatrixXd> &es, Eigen::MatrixXd &m, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double, 3> &D)
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
void Dwp_ba::cherenkov(Eigen::EigenSolver<Eigen::MatrixXd> &es, Eigen::MatrixXd &m, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double, 3> &D)
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

// normal Doppler shift  n=1 : \omega - k_\parallel v_\parallel - \Omega_e/\gamma = 0
void Dwp_ba::normal(Eigen::EigenSolver<Eigen::MatrixXd> &es, Eigen::MatrixXd &m, Eigen::VectorXcd &ev, double &v, double &xi, std::array<double, 3> &D)
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

void Dwp_ba::setup(double w0_, double k10_, double k20_, double dk10_, double wpe)
{
	double vip12, xijp12, gm, tmp, xi;
	int i, j, idx;

	w0=w0_; k10=k10_; k20=k20_; dk10=dk10_; wpe2 = wpe*wpe;

	std::array<double, 3> D_tmp;

	PetscPrintf(PETSC_COMM_WORLD,"Setting up the bounce-averaged wave-particle diffusion operator\n");

	pp.resize((my_mesh->xm +2)*(my_mesh->ym+2));
	pxi.resize((my_mesh->xm +2)*(my_mesh->ym+2));
	xixi.resize((my_mesh->xm +2)*(my_mesh->ym+2));

	Eigen::MatrixXd m(4, 4);
	Eigen::EigenSolver<Eigen::MatrixXd> es;
	Eigen::VectorXcd ev(4);

    Eigen::MatrixXd mD(2,2);
	Eigen::EigenSolver<Eigen::MatrixXd> es2(mD);
	Eigen::VectorXcd ev2(2);

	double kappa, theta, dtheta;
	int nsub=50;  // intervals in poloidal angle


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

		if (xijp12*xijp12 >= 1.)
			kappa = 1.e4;
		else
			kappa = 1. + (xijp12*xijp12/(xic*xic) - 1.)/(1. - xijp12*xijp12);

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

			if (kappa >=1. ) { // passing
				dtheta = PI/nsub;

                //we integrate from -Pi to Pi because D(p,xi) may not be symmatric
				for(int k=-nsub; k<nsub; k++)
				{
					theta = (k+0.5)*dtheta;
					xi = 1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xijp12*xijp12); // pitch-angle at theta
					xi =sqrt(fabs(xi));
					(xijp12>=0)? xi*=1.: xi*=-1.; //xi and xi0 always have to same sign in the passing region

					D_tmp.fill(0.);
//					anomalous(es, m, ev, vip12, xi, D_tmp);
//					cherenkov(es, m, ev, vip12, xi, D_tmp);
					normal(es, m, ev, vip12, xi, D_tmp);

					(xi == 0)? tmp = 0. : tmp=((1.-eps*cos(theta))/(1.-eps)* xijp12/xi);
					D_tmp[0] *= tmp;
					(tmp == 0)? D_tmp[2]=0 : D_tmp[2]*=(1./tmp);

					tmp = dtheta/(PI*2.);
					pp[idx] += D_tmp[0] * tmp;
					pxi[idx] += D_tmp[1] * tmp;
					xixi[idx] += D_tmp[2] * tmp;
				}

			}  else {// trapped
				dtheta = 2.*asin(kappa)/nsub;

				for(int k=-nsub; k<nsub; k++)
				{
					theta = (k+0.5)*dtheta;
					xi = (1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xijp12*xijp12)); // pitch-angle at theta
					xi =sqrt(fabs(xi));
					(xijp12>=0)? xi*=1.: xi*=-1.;

					D_tmp.fill(0.);
//					anomalous(es, m, ev, vip12, xi, D_tmp);
//					cherenkov(es, m, ev, vip12, xi, D_tmp);
					normal(es, m, ev, vip12, xi, D_tmp);

					(xi == 0)? tmp=0 : tmp=((1.-eps*cos(theta))/(1.-eps) * xijp12/xi);
					D_tmp[0] *= tmp;
					(tmp == 0)? D_tmp[2]*=0 : D_tmp[2] *= (1./tmp);

					tmp = dtheta/(PI*2.);  // the two counts the [-theta0, 0] due to symmetry
					pp[idx] += D_tmp[0] * tmp;
					pxi[idx] += D_tmp[1] * tmp;
					xixi[idx] += D_tmp[2] * tmp;

					// for the opposite half of the bounce orbit
					xi = -xi;  // pitch-angle flips sign

					D_tmp.fill(0.);
//					anomalous(es, m, ev, vip12, xi, D_tmp);
//					cherenkov(es, m, ev, vip12, xi, D_tmp);
					normal(es, m, ev, vip12, xi, D_tmp);

					(xi == 0)? tmp = 0 : tmp=((1.-eps*cos(theta))/(1.-eps) * xijp12/xi);
					D_tmp[0] *= tmp;
					(tmp == 0)? D_tmp[2]*=0 : D_tmp[2]*=(1./tmp);

					tmp = -dtheta/(PI*2.); // opposite integration direction in theta flips the sign
					pp[idx] += D_tmp[0] * tmp;
					pxi[idx] += D_tmp[1] * tmp;
					xixi[idx] += D_tmp[2] * tmp;
				}
			}

            if (false)
            {
                //check if D is positive definite; note that eigenvalues of D is real
                mD<<pp[idx] ,   pxi[idx],
                    pxi[idx],  xixi[idx];
                es2.compute(mD, false);
                ev2=es2.eigenvalues();
                if ( std::real(ev2[0])<=0.0 && std::real(ev2[1])<=0.0 )
                {
                    PetscPrintf(PETSC_COMM_WORLD,"ERROR: D has negative eigen at i=%i, j=%i\n",i,j);
                    abort();
                }
            }

           if (i!=my_mesh->xs-1 && i!=my_mesh->xs+my_mesh->xm && j!=my_mesh->ys-1 && j!=my_mesh->ys+my_mesh->ym) {
               
               pp_=pp[idx];
               pxi_=pxi[idx];
               xixi_=xixi[idx];

               if (false && i==0 && j==0)
                        PetscPrintf(PETSC_COMM_WORLD,"i=%i, j=%i, idx=%i, pp=%e\n",i,j, idx, pp[idx]);

               //note that i j is already a global index
			   IDX = i + j*my_mesh->Mx; // index of the diffusion coefficient matrix

               ierr=VecSetValues(PP,  1,&IDX,&pp_,ADD_VALUES); assert(ierr==0);
               ierr=VecSetValues(PXI ,1,&IDX,&pxi_,ADD_VALUES); assert(ierr==0);
               ierr=VecSetValues(XIXI,1,&IDX,&xixi_,ADD_VALUES); assert(ierr==0);
           }

		}//end of p
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


            

    if (false)
    {
        std::ofstream ofile;
        ofile.open("Dwp.dat");
        if (ofile.is_open()) {
	        if (ofile.is_open()) {
                int xm_=my_mesh->xm;
                int ym_=my_mesh->ym;
                //ofile.write((char*)pp.data(), (xm_)*(ym_)*sizeof(double));
	        	for (j=my_mesh->ys; j<my_mesh->ys+my_mesh->ym; j++) { //xi

	        		for (i=my_mesh->xs; i<my_mesh->xs+my_mesh->xm; i++) { //p


	        			idx = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+2); // index of the diffusion coefficient matrix

                        if (i==0 && j==0)
                            PetscPrintf(PETSC_COMM_WORLD,"i=%i, j=%i, idx=%i, pp=%e\n",i,j, idx, pp[idx]);

	        			ofile <<rank<<" "<<i<<" "<<j<<" "<<i+j*my_mesh->Mx<<" "<<pp[idx]<<" "<<pxi[idx]<<" "<<xixi[idx]<<std::endl;
	        			//ofile <<pp[idx]<<" ";
	        		}
                    //ofile << std::endl;

	        	}
	        }
        }
        ofile.close();
    }

    PetscPrintf(PETSC_COMM_WORLD,"Setting up wpi whole domain finished\n");
	  ppFace.resize((my_mesh->xm +1)*2);
	 pxiFace.resize((my_mesh->xm +1)*2);
	xixiFace.resize((my_mesh->xm +1)*2);

    double xijp1;
    for (j=0;j<2;j++){
        //note it is passing zone based on the defination
        if (j==0)
            xijp1=my_mesh->yf[my_mesh->My2];
        else
            xijp1=my_mesh->yf[my_mesh->My1];

	    for (i=my_mesh->xs-1; i<my_mesh->xs+my_mesh->xm; i++) { //p

			idx = (i-my_mesh->xs+1) + j*(my_mesh->xm+1); // index of the diffusion coefficient matrix

	    	  ppFace[idx] = 0;
	    	 pxiFace[idx] = 0;
	    	xixiFace[idx] = 0;

			if (i<0)
				vip12 = 2.*my_mesh->xmin - my_mesh->xx[0];
			else if (i==my_mesh->Mx-1)
				vip12 = 2.*my_mesh->xmax - my_mesh->xx[my_mesh->Mx-1];
			else
				vip12 = my_mesh->xx[i];

	    	dtheta = PI/nsub;

            //passing
	    	for(int k=-nsub; k<nsub; k++)
	    	{
	    		theta = (k+0.5)*dtheta;
	    		xi = (1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xijp1*xijp1)); // pitch-angle at theta
                xi = sqrt(fabs(xi));
	    		(xijp1>=0)? xi*=1.: xi*=-1.;

	    		D_tmp.fill(0.);
	    		normal(es, m, ev, vip12, xi, D_tmp);

	    		(xi == 0)? tmp = 0. : tmp=((1.-eps*cos(theta))/(1.-eps)* xijp1/xi);
	    		D_tmp[0] *= tmp;
	    		(tmp == 0)? D_tmp[2]*=0 : D_tmp[2]*=(1./tmp);

	    		tmp = dtheta/(PI*2.);
	    		  ppFace[idx] += D_tmp[0] * tmp;
	    		 pxiFace[idx] += D_tmp[1] * tmp;
	    		xixiFace[idx] += D_tmp[2] * tmp;
	    	}

	    }//end of p
    }

    PetscPrintf(PETSC_COMM_WORLD,"Setting up finished\n");
};

//setupFace is not right: we need two sets of pp pxi xixi in the two fluxes
//I run into some strange malloc bugs and just gave up... -QT
//The current version just uses a linear extrapolation instead
void Dwp_ba::setupFace(double w0_, double k10_, double k20_, double dk10_, double wpe)
{
    double vip1, xijp1, gm, tmp, xi;
	int i, j, idx;

	w0=w0_; k10=k10_; k20=k20_; dk10=dk10_; wpe2 = wpe*wpe;

	std::array<double, 3> D_tmp;

	PetscPrintf(PETSC_COMM_WORLD,"Setting up the bounce-averaged wave-particle diffusion operator (cell-face values)\n");

	pp.resize((my_mesh->xm +1)*(my_mesh->ym+1));
	pxi.resize((my_mesh->xm +1)*(my_mesh->ym+1));
	xixi.resize((my_mesh->xm +1)*(my_mesh->ym+1));

	Eigen::MatrixXd m(4, 4);
	Eigen::EigenSolver<Eigen::MatrixXd> es;
	Eigen::VectorXcd ev(4);

    Eigen::MatrixXd mD(2,2);
	Eigen::EigenSolver<Eigen::MatrixXd> es2(mD);
	Eigen::VectorXcd ev2(2);

	double kappa, theta, dtheta;
	int nsub=50;  // intervals in poloidal angle

	// compute the wave-particle diffusion matrix on cell centers, values outside the domain will not be used
	for (j=my_mesh->ys-1; j<my_mesh->ys+my_mesh->ym; j++) { //xi

		if (j < my_mesh->My2)
			xijp1 = my_mesh->yf[j];   // cell upper face
		else
			xijp1 = (j<(my_mesh->My-1)) ? (my_mesh->yf[j+1]) : (1.0);

		if (xijp1*xijp1 >= 1.)
			kappa = 1.e4;
		else
			kappa = 1. + (xijp1*xijp1/(xic*xic) - 1.)/(1. - xijp1*xijp1);

		for (i=my_mesh->xs-1; i<my_mesh->xs+my_mesh->xm; i++) { //p

			idx = (i-my_mesh->xs+1) + (j-my_mesh->ys+1)*(my_mesh->xm+1); // index of the diffusion coefficient matrix
			pp[idx] = 0;
			pxi[idx] = 0;
			xixi[idx] = 0;

			if (i == -1 )
				vip1 = my_mesh->xmin;
			else
				vip1 =  my_mesh->xf[i];

			if (kappa >=1. ) { // passing
				dtheta = PI/nsub;

				for(int k=0; k<nsub; k++)
				{
					theta = (k+0.5)*dtheta;
					xi = (1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xijp1*xijp1)); // pitch-angle at theta
					(xi>0)? xi =sqrt(xi) : xi=0.;
					(xijp1>=0)? xi*=1.: xi*=-1.;

					D_tmp.fill(0.);
					normal(es, m, ev, vip1, xi, D_tmp);

					(xi == 0)? tmp = 0. : tmp=((1.-eps*cos(theta))/(1.-eps)* xijp1/xi);
					D_tmp[0] *= tmp;
					(tmp == 0)? D_tmp[2]*=0 : D_tmp[2]*=(1./tmp);

					tmp = dtheta/(PI);
					pp[idx] += D_tmp[0] * tmp;
					pxi[idx] += D_tmp[1] * tmp;
					xixi[idx] += D_tmp[2] * tmp;
				}

			}  else {// trapped
				dtheta = (0.98*2.*asin(kappa))/nsub;

				for(int k=0; k<nsub; k++)
				{
					theta = (k+0.5)*dtheta;
					xi = (1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xijp1*xijp1)); // pitch-angle at theta
					(xi>0)? xi =sqrt(xi) : xi=0.;
					(xijp1>=0)? xi*=1.: xi*=-1.;

					D_tmp.fill(0.);
					normal(es, m, ev, vip1, xi, D_tmp);

					(xi == 0)? tmp=0 : tmp=((1.-eps*cos(theta))/(1.-eps) * xijp1/xi);
					D_tmp[0] *= tmp;
					(tmp == 0)? D_tmp[2]*=0 : D_tmp[2] *= (1./tmp);

					tmp = dtheta/(PI);  // the two counts the [-theta0, 0] due to symmetry
					pp[idx] += D_tmp[0] * tmp;
					pxi[idx] += D_tmp[1] * tmp;
					xixi[idx] += D_tmp[2] * tmp;

					// for the opposite half of the bounce orbit
					xi = -xi;  // pitch-angle flips sign

					D_tmp.fill(0.);
					normal(es, m, ev, vip1, xi, D_tmp);

					(xi == 0)? tmp = 0 : tmp=((1.-eps*cos(theta))/(1.-eps) * xijp1/xi);
					D_tmp[0] *= tmp;
					(tmp == 0)? D_tmp[2]*=0 : D_tmp[2]*=(1./tmp);

					tmp = -dtheta/(PI); // opposite integration direction in theta flips the sign
					pp[idx] += D_tmp[0] * tmp;
					pxi[idx] += D_tmp[1] * tmp;
					xixi[idx] += D_tmp[2] * tmp;
				}
			}

            if (false){
                //check if D is positive definite; note that eigenvalues of D is real
                mD<<pp[idx] ,   pxi[idx],
                    pxi[idx],  xixi[idx];
                es2.compute(mD, false);
                ev2=es2.eigenvalues();
                if ( std::real(ev2[0])<=0.0 && std::real(ev2[1])<=0.0 )
                {
                    PetscPrintf(PETSC_COMM_WORLD,"ERROR: D has negative eigen at i=%i, j=%i\n",i,j);
                    MPI_Barrier(PETSC_COMM_WORLD);
                    abort();
                }
            }
		}//end of p
	}

	  ppFace.resize((my_mesh->xm +1)*2);
	 pxiFace.resize((my_mesh->xm +1)*2);
	xixiFace.resize((my_mesh->xm +1)*2);

    for (j=0;j<2;j++){
        //note it is passing zone based on the defination
        if (j==0)
            xijp1=my_mesh->yf[my_mesh->My2];
        else
            xijp1=my_mesh->yf[my_mesh->My1];

	    for (i=my_mesh->xs-1; i<my_mesh->xs+my_mesh->xm; i++) { //p

			idx = (i-my_mesh->xs+1) + j*(my_mesh->xm+1); // index of the diffusion coefficient matrix

	    	  ppFace[idx] = 0;
	    	 pxiFace[idx] = 0;
	    	xixiFace[idx] = 0;

	    	if (i == -1)
	    		vip1 = my_mesh->xmin;
	    	else
	    		vip1 =  my_mesh->xf[i];

	    		dtheta = PI/nsub;

                //passing
	    		for(int k=0; k<nsub; k++)
	    		{
	    			theta = (k+0.5)*dtheta;
	    			xi = (1. - (1. - eps*cos(theta))/(1. - eps) * (1.-xijp1*xijp1)); // pitch-angle at theta
	    			(xi>0)? xi =sqrt(xi) : xi=0.;
	    			(xijp1>=0)? xi*=1.: xi*=-1.;

	    			D_tmp.fill(0.);
	    			normal(es, m, ev, vip1, xi, D_tmp);

	    			(xi == 0)? tmp = 0. : tmp=((1.-eps*cos(theta))/(1.-eps)* xijp1/xi);
	    			D_tmp[0] *= tmp;
	    			(tmp == 0)? D_tmp[2]*=0 : D_tmp[2]*=(1./tmp);

	    			tmp = dtheta/(PI);
	    			  ppFace[idx] += D_tmp[0] * tmp;
	    			 pxiFace[idx] += D_tmp[1] * tmp;
	    			xixiFace[idx] += D_tmp[2] * tmp;
	    		}

	    }//end of p
    }

};
