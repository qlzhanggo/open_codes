
static char help[] = "This program solves the Relativistic Fokker-Planck equation";

/*---------------------------------------------------------------------
    Program usage example:  mpiexec -n <procs> ./runaway -file <init data> -dt 0.0001 -skip 0.5 -ftime 50 -fdcoloring -ts_type arkimex -ts_arkimex_type l2
  ------------------------------------------------------------------------- */

#include <petsc.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscdmshell.h>
#include <petscsys.h>
#include <assert.h>
#include <time.h>
#include <memory>
#include <string.h>

#include "fk_equ/userdata.h"
#include "fk_equ/fk_equ.h"
#include "fk_equ/slab/fk_equ_slab.h"
#include "fk_equ/toroidal/fk_equ_ba.h"

#include "fk_equ/slab/knockon_none.h"
#include "fk_equ/slab/knockon_rp.h"
#include "fk_equ/slab/knockon_chiu.h"
#include "fk_equ/toroidal/knockon_none_ba.h"
#include "fk_equ/toroidal/knockon_rp_ba.h"
#include "fk_equ/toroidal/knockon_chiu_ba.h"

#include "field/field_simple.h"

#include "simulation.h"
#include "mesh.h"

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;

  PetscInitialize(&argc, &argv, (char *)0, help);

  RegisterNewARKMethods();
  PetscPrintf(PETSC_COMM_WORLD, "%s \n", help);
  std::unique_ptr<mesh> my_mesh(new mesh("mesh.m"));
  std::unique_ptr<Field_EQU> Efield(new field_simple());
  std::unique_ptr<fk_equ>  my_equ; 

  int count = 1;
  while (std::strcmp(argv[count], "-fk_type") !=0 && count<argc)
    count++;

  if (count<argc-1) {
    PetscPrintf(PETSC_COMM_WORLD, "Setting fk_type to %s\n", argv[count+1]);
    if ( std::strcmp(argv[count+1],"slab") ==0 ) 
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_slab<knockon_none>(my_mesh.get(), Efield.get(), "input_params.m"));
    else if ( std::strcmp(argv[count+1], "bounce")==0 )
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_ba<knockon_none_ba>(my_mesh.get(), Efield.get(), "input_params.m"));
    else if ( std::strcmp(argv[count+1], "slab_chiu")==0 )
      my_equ = std::unique_ptr<fk_equ>( new fk_equ_slab<knockon_chiu>(my_mesh.get(), Efield.get(), "input_params.m"));
    else if ( std::strcmp(argv[count+1], "slab_rp")==0 )
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_slab<knockon_rp>(my_mesh.get(), Efield.get(), "input_params.m"));
    else if ( std::strcmp(argv[count+1], "bounce_rp")==0 )
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_ba<knockon_rp_ba>(my_mesh.get(), Efield.get(), "input_params.m"));
    else if ( std::strcmp(argv[count+1], "bounce_chiu")==0 )
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_ba<knockon_chiu_ba>(my_mesh.get(), Efield.get(), "input_params.m"));
    else {
      PetscPrintf(PETSC_COMM_WORLD, "The type %s does not match any fk_type, using the default type slab with no avalanche\n", argv[count+1]);
      my_equ = std::unique_ptr<fk_equ>(new fk_equ_slab<knockon_none>(my_mesh.get(), Efield.get(), "input_params.m"));
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Setting fk_type to standard primary in slab geometry.\n");
    my_equ = std::unique_ptr<fk_equ>(new fk_equ_slab<knockon_none>(my_mesh.get(), Efield.get(), "input_params.m"));
  }

  Simulation sim(my_mesh.get(), my_equ.get(), Efield.get());

  sim.solve();
  sim.cleanup();

  my_mesh.reset();
  Efield.reset();
  my_equ.reset();

  ierr = PetscFinalize();

  return 0;
}
