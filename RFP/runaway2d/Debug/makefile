HOST=$(shell hostname)
RM := rm -rf

ifneq ($(findstring mavis,$(HOST)), )
   $(info The host is $(HOST))
   PETSC_DIR := /apps/contrib/petsc-3.8.2
   PETSC_ARCH := linux-gnu
   MPI_DIR := /apps/mpi/mpich/3.2
   BOOST_DIR := /apps/contrib/boost_1_65_1
   EIGEN_DIR := /apps/contrib/eigen
   CSTD := c++11

else ifneq ($(findstring pn18,$(HOST)), )
   $(info The host is $(HOST))
   PETSC_DIR := $(HOME)/software/petsc-3.16.5
   PETSC_ARCH := macx
   MPI_DIR := $(HOME)/.brew
   BOOST_DIR := $(HOME)/software/boost_1_69_0
   EIGEN_DIR := $(HOME)/software/eigen
   CSTD := c++11
else ifneq ($(findstring gr-fe,$(HOST)), )
   $(info The host is $(HOST))
   PETSC_DIR := $(HOME)/Software/petsc-3.11.1
   PETSC_ARCH := arch-linux2-c-opt
   MPI_DIR := /usr/projects/hpcsoft/toss3/grizzly/openmpi/2.1.2-gcc-9.3.0
   BOOST_DIR := $(HOME)/scratch/Software/boost
   EIGEN_DIR := $(HOME)/scratch/Software/eigen
   CSTD := c++11
endif

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# All of the sources participating in the build are defined here
SUBDIRS := src src/fk_equ src/fk_equ/slab src/fk_equ/toroidal src/field

-include src/subdir.mk

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: runaway

# Tool invocations
runaway: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: MPIC++ Linker'
	$(MPI_DIR)/bin/mpic++ -std=$(CSTD) -w -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -o "runaway" $(OBJS) $(USER_OBJS) $(PETSC_TS_LIB)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
myclean:
	-$(RM) $(OBJS) $(C++_DEPS) $(EXECUTABLES) $(CC_DEPS) $(C_DEPS) $(CPP_DEPS) $(CXX_DEPS) $(C_UPPER_DEPS) runaway
	-@echo ' '

.PHONY: all myclean dependents
.SECONDARY:

-include ../makefile.targets
