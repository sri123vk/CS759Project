CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra
CPPFLAGS  += -Iinclude
NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -Wall,-Wextra

BIN := bin
CPU_SRCS := src/main.cpp src/conv2d_seq.cpp
TEST_SRCS := tests/test_conv.cpp src/conv2d_seq.cpp
HEADERS := include/conv2d.h include/utils.h

# The OpenMP version is built in (and tested) once its source file exists.
ifneq ($(wildcard src/conv2d_omp.cpp),)
  CPU_SRCS  += src/conv2d_omp.cpp
  TEST_SRCS += src/conv2d_omp.cpp
  OMPFLAGS  := -fopenmp -DCONV_HAVE_OMP
endif

# The CUDA version is built in when nvcc is on the PATH; `make CUDA=0` skips it.
CUDA ?= $(if $(shell command -v $(NVCC) 2>/dev/null),1,0)
ifeq ($(CUDA),1)
  CUDA_OBJ    := $(BIN)/conv2d_cuda.o
  CUDAFLAGS   := -DCONV_HAVE_CUDA
  CUDA_LIBDIR ?= $(abspath $(dir $(shell command -v $(NVCC)))../lib64)
  CUDA_LIBS   := -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart
endif

# External check (make validate): bin/conv dumps its output for each shape
# and implementation, then validation/validate.py compares every output with
# SciPy (or PyTorch). Shapes are HxWxK. Implementations default to seq, plus
# cuda when it is built in.
#
# `make test` runs it after the unit tests. VALIDATE=auto (default) runs it
# if $(PYTHON) has numpy and scipy (or torch) and otherwise prints SKIPPED;
# VALIDATE=1 runs it regardless, so a missing library fails the build;
# VALIDATE=0 skips it.
VALIDATE        ?= auto
PYTHON          ?= python3
VALIDATE_DIR    ?= data/validate
VALIDATE_SHAPES ?= 2048x2048x3 2048x2048x5 2048x2048x7 1531x2777x7
VALIDATE_IMPLS  ?= seq $(if $(filter 1,$(CUDA)),cuda)

.PHONY: all test validate clean

all: $(BIN)/conv

$(BIN)/conv: $(CPU_SRCS) $(CUDA_OBJ) $(HEADERS) | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(CUDAFLAGS) $(CPU_SRCS) $(CUDA_OBJ) $(CUDA_LIBS) -o $@

$(BIN)/test_conv: $(TEST_SRCS) $(CUDA_OBJ) $(HEADERS) | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(CUDAFLAGS) $(TEST_SRCS) $(CUDA_OBJ) $(CUDA_LIBS) -o $@

$(BIN)/conv2d_cuda.o: src/conv2d_cuda.cu include/conv2d.h | $(BIN)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

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
	  H=$${s%%x*}; rest=$${s#*x}; W=$${rest%%x*}; K=$${rest#*x}; \
	  for impl in $(VALIDATE_IMPLS); do \
	    echo "  $$impl $${H}x$${W} K=$$K"; \
	    ./$(BIN)/conv --impl $$impl --H $$H --W $$W --K $$K \
	      --iters 1 --warmup 0 --dump $(VALIDATE_DIR) > /dev/null; \
	  done; \
	done
	$(PYTHON) validation/validate.py $(VALIDATE_DIR)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
