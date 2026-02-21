package tensorrt

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lcudart -lnvinfer -lstdc++
#cgo CXXFLAGS: -I/usr/local/cuda/include -I/usr/lib/x86_64-linux-gnu/include -std=c++17 -Wall -O3
#include <stdlib.h>
#include "wrapper.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

// EngineContext is an opaque handle to the C++ EngineContext
type EngineContext struct {
	ctx C.EngineContext
}

// LoadEngine loads a TensorRT engine from a file and returns an EngineContext
func LoadEngine(enginePath string, gpuID int) (*EngineContext, error) {
	cEnginePath := C.CString(enginePath)
	defer C.free(unsafe.Pointer(cEnginePath))

	ctx := C.LoadEngine(cEnginePath, C.int(gpuID))
	if ctx == nil {
		return nil, errors.New("failed to load TensorRT engine")
	}
	return &EngineContext{ctx: ctx}, nil
}

// BuildEngineFromONNX builds a TensorRT engine from an ONNX model
func BuildEngineFromONNX(onnxPath, enginePath string) error {
	cOnnxPath := C.CString(onnxPath)
	cEnginePath := C.CString(enginePath)
	defer C.free(unsafe.Pointer(cOnnxPath))
	defer C.free(unsafe.Pointer(cEnginePath))

	res := C.BuildEngineFromONNX(cOnnxPath, cEnginePath)
	if res != 0 {
		return errors.New("failed to build TensorRT engine from ONNX")
	}
	return nil
}

// Free releases the TensorRT engine context memory
func (e *EngineContext) Free() {
	if e.ctx != nil {
		C.FreeEngine(e.ctx)
		e.ctx = nil
	}
}

// RunInference executes the model. For scaffolding this just passes buffers.
func (e *EngineContext) RunInference(input []float32, output []float32, width int, height int) error {
	if e.ctx == nil {
		return errors.New("engine context is nil")
	}

	if len(input) == 0 || len(output) == 0 {
		return errors.New("input and output buffers must not be empty")
	}

	// For a real integration we'd handle allocating cuda memory inside the wrapper,
	// but here we just pass our float buffers to the C++ logic.
	cInput := (*C.float)(unsafe.Pointer(&input[0]))
	cOutput := (*C.float)(unsafe.Pointer(&output[0]))
	inputSize := C.int(len(input))
	outputSize := C.int(len(output))
	cWidth := C.int(width)
	cHeight := C.int(height)

	res := C.RunInference(e.ctx, cInput, cOutput, inputSize, outputSize, cWidth, cHeight)
	if res != 0 {
		return errors.New("inference failed")
	}
	return nil
}
