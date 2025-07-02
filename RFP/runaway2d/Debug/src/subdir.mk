# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/fk_equ/fk_equ.cpp \
../src/fk_equ/slab/fk_equ_slab.cpp \
../src/fk_equ/toroidal/fk_equ_ba.cpp \
../src/fk_equ/slab/knockon_rp.cpp \
../src/fk_equ/slab/knockon_chiu.cpp \
../src/fk_equ/slab/Dwp_slab.cpp \
../src/fk_equ/toroidal/knockon_rp_ba.cpp \
../src/fk_equ/toroidal/knockon_chiu_ba.cpp \
../src/fk_equ/toroidal/Dwp_ba.cpp \
../src/simulation.cpp \
../src/main.cpp \

OBJS += \
./src/fk_equ/fk_equ.o \
./src/fk_equ/slab/fk_equ_slab.o \
./src/fk_equ/toroidal/fk_equ_ba.o \
./src/fk_equ/slab/knockon_rp.o \
./src/fk_equ/slab/knockon_chiu.o \
./src/fk_equ/slab/Dwp_slab.o \
./src/fk_equ/toroidal/knockon_rp_ba.o \
./src/fk_equ/toroidal/knockon_chiu_ba.o \
./src/fk_equ/toroidal/Dwp_ba.o \
./src/simulation.o \
./src/main.o \

CPP_DEPS += \
./src/fk_equ/fk_equ.d \
./src/fk_equ/slab/fk_equ_slab.d \
./src/fk_equ/toroidal/fk_equ_ba.d \
./src/fk_equ/slab/knockon_rp.d \
./src/fk_equ/slab/knockon_chiu.d \
./src/fk_equ/slab/Dwp_slab.d \
./src/fk_equ/toroidal/knockon_rp_ba.d \
./src/fk_equ/toroidal/knockon_chiu_ba.d \
./src/fk_equ/toroidal/Dwp_ba.d \
./src/simulation.d \
./src/main.d \

dir_guard=@mkdir -p $(@D)

# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	$(dir_guard)
	@echo 'Building file: $<'
	@echo 'Invoking: MPIC++ Compiler'
	$(MPI_DIR)/bin/mpic++ -std=$(CSTD) -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(MPI_DIR)/include -I$(BOOST_DIR) -I$(EIGEN_DIR) -O3 -w -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/fk_equ/%.o: ../src/fk_equ/%.cpp
	$(dir_guard)
	@echo 'Building file: $<'
	@echo 'Invoking: MPIC++ Compiler'
	$(MPI_DIR)/bin/mpic++ -std=$(CSTD) -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(MPI_DIR)/include -I$(BOOST_DIR) -O3 -w -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/fk_equ/slab/%.o: ../src/fk_equ/slab/%.cpp
	$(dir_guard)
	@echo 'Building file: $<'
	@echo 'Invoking: MPIC++ Compiler'
	$(MPI_DIR)/bin/mpic++ -std=$(CSTD) -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(MPI_DIR)/include -I$(BOOST_DIR) -I$(EIGEN_DIR) -O3 -w -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/fk_equ/toroidal/%.o: ../src/fk_equ/toroidal/%.cpp
	$(dir_guard)
	@echo 'Building file: $<'
	@echo 'Invoking: MPIC++ Compiler'
	$(MPI_DIR)/bin/mpic++ -std=$(CSTD) -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(MPI_DIR)/include -I$(BOOST_DIR) -I$(EIGEN_DIR) -O3 -w -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
