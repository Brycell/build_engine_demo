//
// Created by cao on 19-10-26.
//

#ifndef CTDET_TRT_CTDETNET_H
#define CTDET_TRT_CTDETNET_H

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <memory>
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

static auto StreamDeleter = [](cudaStream_t* pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };
    

enum class RUN_MODE
{
    FLOAT32 = 0 ,
    FLOAT16 = 1 ,
    INT8    = 2
};

class Net
{
public:
    Net();
    ~Net();
    void InitEngine(const std::string& onnxFile, const std::string& dir_path, RUN_MODE mode = RUN_MODE::FLOAT32);
    void BuildEngine();
    void SaveEngine(const std::string& fileName);
private:
    inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
    {
        std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
        if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
        {
            pStream.reset(nullptr);
        }

        return pStream;
    }
    nvinfer1::IBuilder* builder_;
    nvinfer1::INetworkDefinition* network_;
    nvinfer1::IBuilderConfig* config_;
    nvonnxparser::IParser* parser_;
    nvinfer1::ICudaEngine* engine_;

};



#endif //CTDET_TRT_CTDETNET_H
