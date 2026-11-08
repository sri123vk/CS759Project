CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
CPPFLAGS += -Iinclude

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

.PHONY: all test clean

all: $(BIN)/conv

$(BIN)/conv: $(CPU_SRCS) $(HEADERS) | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(CPU_SRCS) -o $@

$(BIN)/test_conv: $(TEST_SRCS) $(HEADERS) | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(TEST_SRCS) -o $@

test: $(BIN)/test_conv
	./$(BIN)/test_conv

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
