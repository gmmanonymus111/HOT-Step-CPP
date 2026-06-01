/*
 * test_trtllm_raw.cpp — Direct TensorRT engine test (bypasses TRT-LLM Executor)
 *
 * Tests if we can load and run the TRT engine file directly using the
 * nvinfer API, which is what we ultimately need for HOT-Step.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <cassert>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <NvInfer.h>
#include <cuda_runtime.h>

// TensorRT logger
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << " Raw TensorRT Engine Test (HOT-Step)" << std::endl;
    std::cout << "=====================================" << std::endl;

    // ========== Stage 1: CUDA Device Info ==========
    std::cout << "\n=== STAGE 1: CUDA Device ===" << std::endl;
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "FAIL: No CUDA devices found" << std::endl;
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "VRAM: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
    
    // ========== Stage 2: Load Engine File ==========
    std::cout << "\n=== STAGE 2: Load Engine ===" << std::endl;
    const std::string enginePath = "test_smoke.engine";
    
    if (!std::filesystem::exists(enginePath)) {
        std::cerr << "FAIL: Engine not found at " << enginePath << std::endl;
        return 1;
    }
    
    auto fileSize = std::filesystem::file_size(enginePath);
    std::cout << "Engine file: " << (fileSize / (1024*1024)) << " MB" << std::endl;
    
    // Read engine into memory
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(enginePath, std::ios::binary);
    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();
    
    auto readMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Read in " << readMs << " ms" << std::endl;
    
    // ========== Stage 3: Deserialize Engine ==========
    std::cout << "\n=== STAGE 3: Deserialize Engine ===" << std::endl;
    TrtLogger logger;
    
    auto runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "FAIL: createInferRuntime returned null" << std::endl;
        return 1;
    }
    std::cout << "OK: Runtime created" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    auto engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    auto deserMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    if (!engine) {
        std::cerr << "FAIL: deserializeCudaEngine returned null" << std::endl;
        std::cerr << "  (Engine may have been built with a different TRT version)" << std::endl;
        delete runtime;
        return 1;
    }
    std::cout << "OK: Engine deserialized in " << deserMs << " ms" << std::endl;
    std::cout << "  Num I/O tensors: " << engine->getNbIOTensors() << std::endl;
    
    // Print I/O tensor names and shapes
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        auto name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto shape = engine->getTensorShape(name);
        auto dtype = engine->getTensorDataType(name);
        
        std::string modeStr = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        std::string shapeStr = "(";
        for (int d = 0; d < shape.nbDims; d++) {
            if (d > 0) shapeStr += ", ";
            shapeStr += std::to_string(shape.d[d]);
        }
        shapeStr += ")";
        
        std::cout << "  [" << i << "] " << modeStr << " \"" << name << "\" " << shapeStr << std::endl;
    }
    
    // ========== Stage 4: Create Execution Context ==========
    std::cout << "\n=== STAGE 4: Execution Context ===" << std::endl;
    auto context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "FAIL: createExecutionContext returned null" << std::endl;
        delete engine;
        delete runtime;
        return 1;
    }
    std::cout << "OK: Execution context created" << std::endl;
    
    // ========== Summary ==========
    std::cout << "\n=====================================" << std::endl;
    std::cout << " ALL STAGES PASSED!" << std::endl;
    std::cout << " TensorRT engine loads and context" << std::endl;
    std::cout << " can be created successfully." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Cleanup
    delete context;
    delete engine;
    delete runtime;
    
    return 0;
}
