CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra
CPPFLAGS  += -Iinclude
NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -Wall,-Wextra

BIN := bin
CPU_SRCS := src/main.cpp src/seq/conv2d_seq.cpp
TEST_SRCS := tests/test_conv.cpp src/seq/conv2d_seq.cpp
HEADERS := include/conv2d.h include/utils.h

# OpenMP variants: omp, omp_simd, omp_blocked, omp_tiled
OMP_VARIANTS := omp omp_simd omp_blocked omp_tiled
OMP_BUILT    := $(strip $(foreach v,$(OMP_VARIANTS),$(if $(wildcard src/omp/conv2d_$(v).cpp),$(v))))
ifneq ($(OMP_BUILT),)
  OMP_SRCS  := $(OMP_BUILT:%=src/omp/conv2d_%.cpp)
  CPU_SRCS  += $(OMP_SRCS)
  TEST_SRCS += $(OMP_SRCS)
  OMPFLAGS  := -fopenmp $(foreach v,$(OMP_BUILT),-DCONV_HAVE_$(shell echo $(v) | tr a-z A-Z))
endif

# The CUDA version is built in when nvcc is on the PATH; `make CUDA=0` skips it.
CUDA ?= $(if $(shell command -v $(NVCC) 2>/dev/null),1,0)
ifeq ($(CUDA),1)
  CUDA_OBJ := $(BIN)/conv2d_cuda.o $(BIN)/conv2d_cuda_variants.o
  CUDAFLAGS := -DCONV_HAVE_CUDA -DCONV_HAVE_CUDA_VARIANTS
  CUDA_LIBDIR ?= $(abspath $(dir $(shell command -v $(NVCC)))../lib64)
  CUDA_LIBS := -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -pthread
endif

# External check (make validate): bin/conv dumps its output for each shape
# and implementation, then validation/validate.py compares every output with
# SciPy (or PyTorch). Shapes are HxWxK, or HxWxKxC for C input channels.
# Implementations default to seq and every OpenMP version that is built,
# plus cuda and cuda_multiout_tiled_const (the variant that combines shared
# tiles, constant memory and two outputs per thread) when CUDA is built in.
# Every implementation runs every shape. The shapes cover small images,
# odd non-square sizes, large filters, and C from 3 up to 64.
#
# `make test` runs it after the unit tests. VALIDATE=auto (default) runs it
# if $(PYTHON) has numpy and scipy (or torch) and otherwise prints SKIPPED;
# VALIDATE=1 runs it regardless, so a missing library fails the build;
# VALIDATE=0 skips it.
VALIDATE        ?= auto
PYTHON          ?= python3
VALIDATE_DIR    ?= data/validate
VALIDATE_SHAPES ?= 2048x2048x3 2048x2048x5 2048x2048x7 1531x2777x7 \
                   16x16x3 31x17x5x4 \
                   1024x768x5x3 257x263x7x16 300x200x11x8 128x128x7x64
VALIDATE_IMPLS  ?= seq $(OMP_BUILT) $(if $(filter 1,$(CUDA)),cuda cuda_multiout_tiled_const)

.PHONY: all test validate clean

all: $(BIN)/conv

$(BIN)/conv: $(CPU_SRCS) $(CUDA_OBJ) $(HEADERS) | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(CUDAFLAGS) $(CPU_SRCS) $(CUDA_OBJ) $(CUDA_LIBS) -o $@

$(BIN)/test_conv: $(TEST_SRCS) $(CUDA_OBJ) $(HEADERS) | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(CUDAFLAGS) $(TEST_SRCS) $(CUDA_OBJ) $(CUDA_LIBS) -o $@

$(BIN)/conv2d_cuda.o: src/cuda/conv2d_cuda.cu include/conv2d.h | $(BIN)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(BIN)/conv2d_cuda_variants.o: src/cuda/conv2d_cuda_variants.cu include/conv2d.h | $(BIN)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -Xcompiler -pthread -c $< -o $@

test: $(BIN)/test_conv
	./$(BIN)/test_conv
ifeq ($(VALIDATE),0)
	@echo "external check skipped (VALIDATE=0)"
else
	@if [ "$(VALIDATE)" = 1 ] || $(PYTHON) -c "import numpy, scipy" 2>/dev/null \
	    || $(PYTHON) -c "import numpy, torch" 2>/dev/null; then \
	  echo; echo "== external check (make validate)"; \
	  $(MAKE) --no-print-directory validate; \
	else \
	  echo; \
	  echo "*** SKIPPED: external check against SciPy (make validate)."; \
	  echo "*** '$(PYTHON)' is missing, or has neither numpy+scipy nor numpy+torch."; \
	  echo "*** Install them, or use VALIDATE=1 to make this an error."; \
	fi
endif

validate: $(BIN)/conv
	mkdir -p $(VALIDATE_DIR)
	rm -f $(VALIDATE_DIR)/*.bin
	@set -e; for s in $(VALIDATE_SHAPES); do \
	  H=$${s%%x*}; rest=$${s#*x}; W=$${rest%%x*}; rest=$${rest#*x}; \
	  K=$${rest%%x*}; C=1; \
	  case $$rest in *x*) C=$${rest#*x};; esac; \
	  for impl in $(VALIDATE_IMPLS); do \
	    echo "  $$impl $${H}x$${W} K=$$K C=$$C"; \
	    ./$(BIN)/conv --impl $$impl --H $$H --W $$W --K $$K --C $$C \
	      --iters 1 --warmup 0 --dump $(VALIDATE_DIR) > /dev/null; \
	  done; \
	done
	$(PYTHON) validation/validate.py $(VALIDATE_DIR)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
