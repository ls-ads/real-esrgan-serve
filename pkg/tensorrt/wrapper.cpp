#include "wrapper.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;
#ifdef __cplusplus
extern "C" {
#endif

// We just put stub implementations for the scaffolding task to ensure CGO links property.

// Wrapper structure to hold the TensorRT execution state
struct TrtContext {
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
};

EngineContext LoadEngine(const char* enginePath, int gpuID) {
    std::cout << "[C++] Loading Engine: " << enginePath << " on GPU " << gpuID << std::endl;
    
    // Set device
    if (cudaSetDevice(gpuID) != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << gpuID << std::endl;
        return nullptr;
    }

    // Read engine file
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open engine file." << std::endl;
        return nullptr;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read engine file." << std::endl;
        return nullptr;
    }

    TrtContext* trt = new TrtContext();
    
    trt->runtime = createInferRuntime(gLogger);
    if (!trt->runtime) {
        delete trt;
        return nullptr;
    }

    trt->engine = trt->runtime->deserializeCudaEngine(buffer.data(), size);
    if (!trt->engine) {
        delete trt->runtime;
        delete trt;
        return nullptr;
    }

    trt->context = trt->engine->createExecutionContext();
    if (!trt->context) {
        delete trt->engine;
        delete trt->runtime;
        delete trt;
        return nullptr;
    }

    std::cout << "[C++] Engine Loaded Successfully." << std::endl;
    return static_cast<EngineContext>(trt);
}

int RunInference(EngineContext ctx, const float* input, float* output, int inputSize, int outputSize, int width, int height) {
    if (!ctx) return 1;
    TrtContext* trt = static_cast<TrtContext*>(ctx);

    // 1. Establish the exact concrete dimension for this invocation
    // The C++ API requires us to explicitly tell the dynamic profile what shape is actually hitting it
    Dims4 inputDims{1, 3, height, width};
    if (!trt->context->setInputShape("input", inputDims)) {
        std::cerr << "[C++] Invalid input shape for the engine. Max dimensions exceeded?" << std::endl;
        return 1;
    }

    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;

    size_t inputBytes = inputSize * sizeof(float);
    size_t outputBytes = outputSize * sizeof(float);

    // Allocate GPU Memory
    if (cudaMalloc(&deviceInput, inputBytes) != cudaSuccess) {
        std::cerr << "[C++] Failed to allocate device memory for input." << std::endl;
        return 1;
    }
    if (cudaMalloc(&deviceOutput, outputBytes) != cudaSuccess) {
        std::cerr << "[C++] Failed to allocate device memory for output." << std::endl;
        cudaFree(deviceInput);
        return 1;
    }

    // Set Tensor Addresses (Engine binding mapping)
    trt->context->setTensorAddress("input", deviceInput);
    trt->context->setTensorAddress("output", deviceOutput);

    // Create CUDA Stream for async execution
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "[C++] Failed to create CUDA stream." << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    // Host -> Device Copy
    if (cudaMemcpyAsync(deviceInput, input, inputBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "[C++] Failed cudaMemcpyAsync HostToDevice." << std::endl;
        goto cleanup;
    }

    // Execute Inference
    if (!trt->context->enqueueV3(stream)) {
        std::cerr << "[C++] TensorRT enqueueV3 failed." << std::endl;
        goto cleanup;
    }

    // Device -> Host Copy
    if (cudaMemcpyAsync(output, deviceOutput, outputBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "[C++] Failed cudaMemcpyAsync DeviceToHost." << std::endl;
        goto cleanup;
    }

    // Wait for the stream to complete
    cudaStreamSynchronize(stream);

    // Cleanup and Success
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 0;

cleanup:
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
}

void FreeEngine(EngineContext ctx) {
    if (!ctx) return;
    
    std::cout << "[C++] Freeing Engine Context..." << std::endl;
    TrtContext* trt = static_cast<TrtContext*>(ctx);
    
    if (trt->context) delete trt->context;
    if (trt->engine) delete trt->engine;
    if (trt->runtime) delete trt->runtime;
    
    delete trt;
}

int BuildEngineFromONNX(const char* onnxPath, const char* enginePath) {
    std::cout << "[C++] Building Engine from ONNX: " << onnxPath << std::endl;

    // 1. Create Builder and Network
    std::unique_ptr<IBuilder> builder(createInferBuilder(gLogger));
    if (!builder) return 1;

    // kEXPLICIT_BATCH is deprecated in TensorRT 10.x and is the default behavior.
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) return 1;

    // 2. Parse ONNX
    std::unique_ptr<IParser> parser(createParser(*network, gLogger));
    if (!parser) return 1;

    if (!parser->parseFromFile(onnxPath, static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file: " << onnxPath << std::endl;
        return 1;
    }

    // 3. Create Builder Config
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return 1;

    // Set memory pool limit (e.g., 12GB workspace for large profiles)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 12ULL << 30);

    // 4. Configure Optimization Profile for Dynamic Shapes
    // The provided ONNX trace statically bakes in 1x3x64x64. We must override the input layer bounds.
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    ITensor* inputTensor = network->getInput(0);
    const char* inputName = inputTensor->getName();
    
    // min: 1x3x64x64, opt: 1x3x512x512, max: 1x3x2048x2048
    profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4{1, 3, 64, 64});
    profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4{1, 3, 512, 512});
    profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4{1, 3, 2048, 2048});
    
    config->addOptimizationProfile(profile);

    // 5. Build Serialized Engine
    std::unique_ptr<IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        std::cerr << "Engine serialization failed. Check if ONNX model ops are supported." << std::endl;
        return 1;
    }

    // 6. Write to File
    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Failed to open engine output file: " << enginePath << std::endl;
        return 1;
    }
    
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    std::cout << "[C++] Engine successfully saved to " << enginePath << std::endl;
    return 0;
}

#ifdef __cplusplus
}
#endif
