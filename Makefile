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

.PHONY: all test clean

all: $(BIN)/conv

$(BIN)/conv: $(CPU_SRCS) $(CUDA_OBJ) $(HEADERS) | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(CUDAFLAGS) $(CPU_SRCS) $(CUDA_OBJ) $(CUDA_LIBS) -o $@

$(BIN)/test_conv: $(TEST_SRCS) $(CUDA_OBJ) $(HEADERS) | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(CUDAFLAGS) $(TEST_SRCS) $(CUDA_OBJ) $(CUDA_LIBS) -o $@

$(BIN)/conv2d_cuda.o: src/conv2d_cuda.cu include/conv2d.h | $(BIN)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

test: $(BIN)/test_conv
	./$(BIN)/test_conv

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
