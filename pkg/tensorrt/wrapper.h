#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* EngineContext;

// Initialize TRT engine from .engine file
EngineContext LoadEngine(const char* enginePath, int gpuID);

// Run inference on the given input, storing result in output pointer.
int RunInference(EngineContext ctx, const float* input, float* output, int inputSize, int outputSize, int width, int height);

// Free the engine context
void FreeEngine(EngineContext ctx);

// Build an engine from ONNX
int BuildEngineFromONNX(const char* onnxPath, const char* enginePath, int useFP16);

#ifdef __cplusplus
}
#endif

#endif // WRAPPER_H
