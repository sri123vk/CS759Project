CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
CPPFLAGS += -Iinclude

BIN := bin
CPU_SRCS := src/main.cpp src/conv2d_seq.cpp

.PHONY: all clean

all: $(BIN)/conv

$(BIN)/conv: $(CPU_SRCS) include/conv2d.h include/utils.h | $(BIN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPU_SRCS) -o $@

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
