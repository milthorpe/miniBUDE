SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

PPWI=4
OPTIONS = 
CHPL_FLAGS=-s PPWI=$(PPWI) --detailed-errors --savec tmp --print-commands
CHPL_SOURCE=Bude.chpl
ifeq ($(CHPL_LOCALE_MODEL),gpu)
# GPU version
ifndef CHPL_GPU
$(error CHPL_GPU not defined)
else
CHPL_FLAGS+=--report-gpu -suseGPU=true --ldflags -no-pie
CHPL_SOURCE+=DeviceDetails.h
ifeq ($(CHPL_GPU),nvidia)
CHPL_SOURCE+=DeviceDetails_cu.o
LDFLAGS+=-L$(CUDA_PATH)/lib64 -lcudart
else
ifeq ($(CHPL_GPU),amd)
CHPL_SOURCE+=DeviceDetails_hip.o
LDFLAGS+=-lamdhip64
else
$(error Unknown value for CHPL_GPU: $(CHPL_GPU))
endif
endif
ifndef CHPL_GPU_ARCH
$(error CHPL_GPU_ARCH not defined)
endif
endif
else
# CPU version
ifdef CHPL_GPU
$(error CHPL_GPU is set, but CHPL_LOCALE_MODEL!= gpu)
endif
endif
ifdef DEBUG
  CHPL_FLAGS+=-g --devel
  CUDACXX_FLAGS=-g
else
  CHPL_FLAGS+=--fast --no-ieee-float# --llvm-remarks=vector
  CUDACXX_FLAGS+=-O3
endif
CHPL = chpl

# -------
EXE = chapel-bude
.PHONY: all $(BENCHMARK_EXE) clean

all: $(EXE)

DeviceDetails_cu.o:	DeviceDetails.cu
ifndef CUDA_PATH
	$(error CUDA_PATH not defined)
endif
	nvcc $(CUDACXX_FLAGS) -c -allow-unsupported-compiler DeviceDetails.cu -o $@

DeviceDetails.hip:	DeviceDetails.cu
ifndef ROCM_PATH
	$(error ROCM_PATH not defined)
endif
	hipify-perl DeviceDetails.cu > DeviceDetails.hip

DeviceDetails_hip.o:	DeviceDetails.hip
ifndef ROCM_PATH
	$(error ROCM_PATH not defined)
endif
	hipcc $(CUDACXX_FLAGS) -c DeviceDetails.hip -o $@

$(EXE): $(CHPL_SOURCE)
	$(CHPL) $(CHPL_FLAGS) -s MINIBUDE_COMPILE_COMMANDS="\"$(CHPL) $(CHPL_FLAGS) $(CHPL_SOURCE) -o $@\"" $(CHPL_SOURCE) -o $@

clean:
	rm -f $(EXE) DeviceDetails_cu.o DeviceDetails_hip.o
