/*
 * mesh.h
 *
 *  Created on: May 11, 2017
 *      Author: zehuag
 */

#ifndef MESH_H_
#define MESH_H_

/*
 * mesh.h
 *
 *  Created on: Sep 23, 2014
 *      Author: zehuag
 */

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>


#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

//#include "userdata.h"

class mesh
{
public:
	double eps;

	int Mx, My;

	double xmin, xmax;
	PetscInt xs, xm, ys, ym;

	std::vector<double> xx, yy; // cell center coordinates
	std::vector<double> xf, yf; // cell face coordinates

	DM da;

	int My1, My2; // trap-passing index 1, trap internal index,
	bool with_trap;

	mesh(char* meshfilename)
	{
		char    foo[30];
		FILE *meshfile = fopen(meshfilename, "r");

		PetscPrintf(PETSC_COMM_WORLD,"read mesh\n");

		if (meshfile == NULL) {
			printf("Unable to open file %s\n", meshfilename);
			exit(1);
		}

		fscanf(meshfile, "%s = %lg;\n" ,foo, &eps);

		//fscanf(meshfile, "%s = %d;\n" ,foo, &Mx);
		//fscanf(meshfile, "%s = %d;\n" ,foo, &My);

		fscanf(meshfile, "%s = %lg;\n" ,foo, &xmin);
		fscanf(meshfile, "%s = %lg;\n" ,foo, &xmax);

		double pc(1.0), dph(0.1), dpl(0.01), xic(-0.8), dxih(0.02), dxil(0.002);
		fscanf(meshfile, "%s = %lg;\n", foo, &pc);
		fscanf(meshfile, "%s = %lg;\n", foo, &dph);
		fscanf(meshfile, "%s = %lg;\n", foo, &dpl);
		fscanf(meshfile, "%s = %lg;\n", foo, &xic);
		fscanf(meshfile, "%s = %lg;\n", foo, &dxih);
		fscanf(meshfile, "%s = %lg;\n", foo, &dxil);

		fclose(meshfile);

		set_mesh(pc, dph, dpl, xic, dxih, dxil);
	};

	void set_mesh(const double& pc, const double& ph, const double& pl, const double& xic, const double& xih, const double& xil)
	{
		int i, j;
		double a, b;
		double pw, xiw;
		double temp;
		double xitp = sqrt(2.*eps/(1.+eps));

		Mx = (xmax-xmin)/pl;
		xf.resize(Mx+1); // initialize the mesh array;
		a = 0.5*(pl+ph); b = 0.5*(ph-pl);
		pw = 1.0; //(0.1*xmax>10.*pl) ? 10.*pl : 0.1*xmax;
		i = 0;
		temp = xmin + (a + b*tanh((xmin - pc)/pw));
		while (temp < xmax) {
			xf[i] = temp;
			temp += (a + b*tanh((temp - pc)/pw));
			i++;
		    //PetscPrintf(PETSC_COMM_WORLD,"%lg, %i, %i\t", temp, i, Mx);
		}
		xf[i] = temp;
		Mx = i+1; xmax = temp;

		My = 2.0/((xil<xih) ? xil:xih);
        My1 = 0;
        My2 = 0;
		yf.resize(My+1); // initialize the mesh array;
		a = 0.5*(xih+xil); b = 0.5*(xih-xil);
		xiw = 0.06; //(0.1>10.*xil) ? 10.*xil : 0.1;
		j = 0;
		temp = -1.0 + (a + b*tanh((-1.0 - xic)/xiw));

		if (xitp>0 && xitp<1) {
            // if non-zero trap region
			while (temp <= -xitp) {
				yf[j] = temp;
				temp += (a + b*tanh((temp - xic)/xiw));
				j++;
			}

			My1 = j-1; // index of the 1st trap-passing boundary
			xitp = - yf[My1]; eps = xitp*xitp/(2.-xitp*xitp); // modifying the xitp, eps according to the mesh

			temp = -xitp + (a + b*tanh((temp - xic)/xiw));
			// inside the trap region
			while (temp < 0) {
				yf[j] = temp;
				temp += (a + b*tanh((temp - xic)/xiw));
				j++;
			}
			yf[j-1] = 0.;
			My2 = j; // index of the 2nd trap-passing boundary
			yf[My2] = xitp; j++;
			temp = -yf[My1-1];
		}

		while (temp < 1.0) {
			yf[j] = temp;
			temp += (a + b*tanh((temp - xic)/xiw));
			j++;
		}// the rest of the pitch-angle region
		My = j;

		xf.resize(Mx);
        yf.resize(My); // resizing to the proper sizes

		if (My2>My1) {
			with_trap = true;

			xx.resize(Mx); yy.resize(My);
			for (i=0; i<Mx; i++)
				xx[i] = get_xc(i);
			for (j=0; j<My; j++)
				yy[j] = get_yc(j);

		} else {
			with_trap = false;
			My1 = -5; My2 = -5;

			My++; // need 1 more point for cell center
			xx.resize(Mx); yy.resize(My);

			for (i=0; i<Mx; i++)
				xx[i] = get_xc(i);

			yy[0] = 0.5*(yf[0] - 1.);
			for (j=1; j<My-1; j++)
				yy[j] = 0.5*(yf[j] + yf[j-1]);
			yy[My-1] = 0.5*(yf[My-2]+1.);
		}

		DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, Mx, My, PETSC_DECIDE, 1, 1, 2, PETSC_NULL, PETSC_NULL, &da);
		DMSetFromOptions(da);
		DMSetUp(da);

		DMDAGetCorners(da, &xs, &ys, PETSC_NULL, &xm, &ym, PETSC_NULL);

		PetscPrintf(PETSC_COMM_WORLD,"-------------------- Mesh parameters ------------------\n");
		PetscPrintf(PETSC_COMM_WORLD,"Grid in p = %d\t", Mx);
		PetscPrintf(PETSC_COMM_WORLD,"Grid in xi = %d\n", My);

        if (false){
            for (i=0; i<Mx; i++){
		        PetscPrintf(PETSC_COMM_WORLD,"%e\t", xx[i]);
            }
		    PetscPrintf(PETSC_COMM_WORLD,"\n\n");
            for (j=0; j<My; j++){
		        PetscPrintf(PETSC_COMM_WORLD,"%e\t", yy[j]);
            }
		    PetscPrintf(PETSC_COMM_WORLD,"\n\n");
        }

		if (with_trap) {
			PetscPrintf(PETSC_COMM_WORLD, "This is a toroidal case:\t");
			PetscPrintf(PETSC_COMM_WORLD, "trap-passing boundary is at xi_c=%lg\n", xitp);
			PetscPrintf(PETSC_COMM_WORLD, "My1 = %d\t", My1);
			PetscPrintf(PETSC_COMM_WORLD, "My2 = %d\n", My2);
		}

		PetscPrintf(PETSC_COMM_WORLD,"pmin = %lg\t", xmin);
		PetscPrintf(PETSC_COMM_WORLD,"pmax = %lg\n", xmax);

		PetscPrintf(PETSC_COMM_WORLD,"--------------------------------------------------------\n");

	}

	double get_yc(int j)
	{
		if (j == 0)
			return 0.5*(-1. + yf[0]);
		else if (j == My-1)
			return 0.5*(yf[j] + 1.);
		else if (j < My2) // xi<0 region
			return 0.5*(yf[j] + yf[j-1]);
		else  // xi>xitp
			return 0.5*(yf[j] + yf[j+1]);
	}

	double get_xc(int i)
	{
		if (i == 0 )
			return 0.5*(xmin + xf[0]);
		else
			return 0.5*(xf[i] + xf[i-1]);
	}

	~mesh()
	{
		DMDestroy(&da);  // da is destroyed here in order to avoid the destructor of mesh object after PetscFinalize
		//TODO: I moved the DM destroy into the simulation class, so it is destroyed before calling PETSCFinanlize in the main code
	}
};

#endif /* MESH_H_ */
