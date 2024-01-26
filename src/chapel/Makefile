SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

WGSIZE = 4
OPTIONS = 
CHPL_FLAGS=--detailed-errors --savec tmp --print-commands
ifeq ($(CHPL_LOCALE_MODEL),gpu)
# GPU version
ifndef CHPL_GPU
$(error CHPL_GPU not defined)
else
CHPL_FLAGS+=-s NUM_TD_PER_THREAD=$(WGSIZE) --report-gpu --ldflags -no-pie
ifeq ($(CHPL_GPU),nvidia)
LDFLAGS+=-L$(CUDA_PATH)/lib64 -lcudart
else
ifeq ($(CHPL_GPU),amd)
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
else
  CHPL_FLAGS+=--fast --no-ieee-float --llvm-remarks=vector
endif
CHPL = chpl

# -------
EXE = chapel-bude
.PHONY: all $(BENCHMARK_EXE) clean

all: $(EXE)

$(EXE): Bude.chpl
	$(CHPL) $(CHPL_FLAGS) Bude.chpl -o $@

clean:
	rm -f $(EXE)
