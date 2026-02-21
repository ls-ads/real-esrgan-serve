.PHONY: build clean run

# Default Cuda and TensorRT paths
CUDA_PATH ?= /usr/local/cuda
TRT_PATH ?= /usr/lib/x86_64-linux-gnu

# CGO flags for compiling and linking
export CGO_CXXFLAGS=-I$(CUDA_PATH)/include -I$(TRT_PATH)/include -O3 -Wall -std=c++17
export CGO_LDFLAGS=-L$(CUDA_PATH)/lib64 -L$(TRT_PATH)/lib -lcudart -lnvinfer -lstdc++

build:
	go build -o real-esrgan-serve main.go

run: build
	./real-esrgan-serve

clean:
	rm -f real-esrgan-serve
