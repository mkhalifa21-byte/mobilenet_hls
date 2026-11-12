# Fast C simulation with plain g++ (no Xilinx tools needed)
#   make        build
#   make run    build and run the testbench
CXX      = g++
CXXFLAGS = -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label
SRC      = tb_mobilenet.cpp mobilenet_ref.cpp mobilenet_kernels.cpp mobilenet_top.cpp

tb: $(SRC) mobilenet.h mobilenet_ref.h
	$(CXX) $(CXXFLAGS) -o tb $(SRC) -lm

run: tb
	./tb

clean:
	rm -f tb

.PHONY: run clean
