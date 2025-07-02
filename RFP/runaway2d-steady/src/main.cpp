static char help[] = "This program solves the steady-state Relativistic Fokker-Planck equation";

/*---------------------------------------------------------------------
    Program usage:  mpirun -n <procs> ./runaway -pc_type mg -ksp_monitor  -snes_view -pc_mg_levels 3 -pc_mg_galerkin -mg_levels_ksp_monitor -snes_monitor -mg_levels_pc_type sor -pc_mg_type full

  ------------------------------------------------------------------------- */

#include <petsc.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscdmshell.h>
#include <petscsys.h>
#include <assert.h>
#include <time.h>
#include <memory>

#include "fk_equ/userdata.h"
#include "fk_equ/fk_equ.h"
#include "fk_equ/slab/fk_equ_slab.h"
#include "fk_equ/toroidal/fk_equ_ba.h"

#include "simulation.h"
#include "mesh.h"

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  PetscPrintf(PETSC_COMM_WORLD, "%s \n", help);

  std::unique_ptr<mesh>  my_mesh(new mesh("mesh.m"));
  std::unique_ptr<fk_equ> my_equ;  //(new fk_equ(my_mesh.get(), "input_params.m"));

  int count = 1;
  while (std::strcmp(argv[count], "-fk_type") !=0 && count<argc)
    count++;

  if (count<argc-1) {
    PetscPrintf(PETSC_COMM_WORLD, "Setting fk_type to %s\n", argv[count+1]);
    if ( std::strcmp(argv[count+1],"slab") ==0 ) 
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_slab(my_mesh.get(), "input_params.m"));
    else if ( std::strcmp(argv[count+1], "bounce")==0 )
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_ba(my_mesh.get(), "input_params.m"));
    else {
      PetscPrintf(PETSC_COMM_WORLD, "The type %s does not match any fk_type, using the default type slab\n", argv[count+1]);
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_slab(my_mesh.get(), "input_params.m"));
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Setting fk_type to standard primary in slab geometry.\n");
    my_equ = std::unique_ptr<fk_equ>(new fk_equ_slab(my_mesh.get(), "input_params.m"));
  }

  Simulation sim(my_mesh.get(), my_equ.get());

  sim.solve();
  sim.cleanup();

  my_mesh.reset();
  my_equ.reset();
  ierr = PetscFinalize();

  return 0;
}
