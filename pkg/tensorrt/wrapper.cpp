#include "wrapper.h"
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

// We just put stub implementations for the scaffolding task to ensure CGO links property.

EngineContext LoadEngine(const char* enginePath, int gpuID) {
    std::cout << "[C++] LoadEngine stub called for " << enginePath << " on GPU " << gpuID << std::endl;
    // In actual implementation, we'll initialize nvinfer1::IRuntime, deserialize, etc.
    // Return a dummy pointer to indicate success for now
    return (void*)0xDEADBEEF;
}

int RunInference(EngineContext ctx, const float* input, float* output, int inputSize, int outputSize) {
    std::cout << "[C++] RunInference stub called" << std::endl;
    // In actual implementation, we'll copy input to device, enqueueV3, copy output to host
    return 0; // 0 for success
}

void FreeEngine(EngineContext ctx) {
    std::cout << "[C++] FreeEngine stub called" << std::endl;
    // Cleanup nvinfer1::IExecutionContext and nvinfer1::ICudaEngine
}

int BuildEngineFromONNX(const char* onnxPath, const char* enginePath) {
    std::cout << "[C++] BuildEngineFromONNX stub called. IN: " << onnxPath << " OUT: " << enginePath << std::endl;
    return 0; // 0 for success
}

#ifdef __cplusplus
}
#endif
