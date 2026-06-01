/*
 * test_trtllm_infer.cpp — Full inference test: load engine, set inputs, run, read outputs
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <cstring>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <NvInfer.h>
#include <cuda_runtime.h>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " TensorRT Full Inference Test (HOT-Step)" << std::endl;
    std::cout << "========================================" << std::endl;

    // ===== Load Engine =====
    const std::string enginePath = "test_smoke.engine";
    std::ifstream file(enginePath, std::ios::binary);
    auto fileSize = std::filesystem::file_size(enginePath);
    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    TrtLogger logger;
    auto runtime = nvinfer1::createInferRuntime(logger);
    auto engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    auto context = engine->createExecutionContext();
    
    std::cout << "Engine loaded: " << engine->getNbIOTensors() << " I/O tensors" << std::endl;

    // ===== Prepare I/O =====
    // Input: [1, 4] float32 — 4 values
    float inputData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    // Output: [1, 8] float32 — 8 values  
    float outputData[8] = {0};

    // Allocate GPU buffers
    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, 4 * sizeof(float));
    cudaMalloc(&d_output, 8 * sizeof(float));

    // Copy input to GPU
    cudaMemcpy(d_input, inputData, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // Bind tensors
    context->setTensorAddress("input", d_input);
    context->setTensorAddress("output_relu", d_output);

    // ===== Run Inference =====
    std::cout << "\nInput:  [";
    for (int i = 0; i < 4; i++) { if (i) std::cout << ", "; std::cout << inputData[i]; }
    std::cout << "]" << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto start = std::chrono::high_resolution_clock::now();
    
    // Run 1000 iterations for timing
    for (int i = 0; i < 1000; i++) {
        bool ok = context->enqueueV3(stream);
        if (!ok && i == 0) {
            std::cerr << "FAIL: enqueueV3 returned false" << std::endl;
            return 1;
        }
    }
    cudaStreamSynchronize(stream);
    
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    // Copy output back
    cudaMemcpy(outputData, d_output, 8 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output: [";
    for (int i = 0; i < 8; i++) { if (i) std::cout << ", "; std::cout << outputData[i]; }
    std::cout << "]" << std::endl;

    // Verify output is not all zeros (ReLU output should have some positive values)
    bool hasNonZero = false;
    for (int i = 0; i < 8; i++) {
        if (outputData[i] != 0.0f) hasNonZero = true;
    }

    std::cout << "\n1000 iterations in " << elapsed << " us (" 
              << (elapsed / 1000.0) << " us/iter)" << std::endl;

    if (hasNonZero) {
        std::cout << "\n========================================" << std::endl;
        std::cout << " ✅ INFERENCE WORKS!" << std::endl;
        std::cout << " Data flows: CPU → GPU → TRT → GPU → CPU" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n⚠️ Output is all zeros (possible but unlikely with random weights)" << std::endl;
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    return hasNonZero ? 0 : 1;
}
