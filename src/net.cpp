#include <assert.h>
#include <fstream>
#include <samples/common/logger.h>
#include <NvInfer.h>

#include "net.h"
#include "EntropyCalibrator.h"

static sample::Logger gLogger;

Net::Net()
{
    engine_ = nullptr;
    builder_ = nullptr;
    network_ = nullptr;
    config_ = nullptr;
}
Net::~Net() 
{
    engine_->destroy();
    config_->destroy();
    parser_->destroy();
    network_->destroy();
    builder_->destroy();

}

void Net::InitEngine(const std::string& onnxFile, const std::string& dir_path, RUN_MODE mode) 
{
    builder_ = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network_ = builder_->createNetworkV2(explicitBatch);

    parser_ = nvonnxparser::createParser(*network_, gLogger);
    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
    if (!parser_->parseFromFile(onnxFile.c_str(), verbosity))
    {
        std::string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }




    int maxBatchSize = 1;
    builder_->setMaxBatchSize(maxBatchSize);
    config_ = builder_->createBuilderConfig();
    config_->setMaxWorkspaceSize(1 << 30);

    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
    if (mode== RUN_MODE::INT8)
    {
        std::cout <<"setInt8Mode"<<std::endl;
        config_->setFlag(nvinfer1::BuilderFlag::kINT8);
        MNISTBatchStream calibrationStream(1, 5, dir_path);
        calibrator.reset(new Int8EntropyCalibrator2<MNISTBatchStream>(calibrationStream, 0, "GroundType"));
        config_->setInt8Calibrator(calibrator.get());  
    }
    else if (mode == RUN_MODE::FLOAT16) 
    {
        std::cout <<"setFp16Mode"<<std::endl;
        config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else
    {
        std::cout <<"setFp32Mode"<<std::endl;
    }

    auto profileStream = makeCudaStream();
    if (!profileStream)
    {
        return;
    } 
    config_->setProfileStream(*profileStream);

    nvinfer1::IOptimizationProfile* profile = builder_->createOptimizationProfile();
    for (uint32_t i = 0, n = network_->getNbInputs(); i < n; i++) {
        // Set formats and data types of inputs
        auto* input = network_->getInput(i);
        input->setType(nvinfer1::DataType::kINT8);
        input->setAllowedFormats(1U << static_cast<int32_t>(nvinfer1::TensorFormat::kLINEAR));
        input->setDynamicRange(-127, 127);

        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 768, 1280));
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 768, 1280));
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 768, 1280));

    }
    for (uint32_t i = 0, n = network_->getNbOutputs(); i < n; i++) {
        // Set formats and data types of inputs
        auto* outnput = network_->getOutput(i);
        outnput->setType(nvinfer1::DataType::kINT32);
        outnput->setAllowedFormats(1U << static_cast<int32_t>(nvinfer1::TensorFormat::kLINEAR));
    }
    config_->addOptimizationProfile(profile);

    std::cout << "Begin building engine..." << std::endl;
    engine_ = builder_->buildEngineWithConfig(*network_, *config_);
    std::cout << "Begin building engine..." << std::endl;
    if (!engine_){
        std::string error_message ="Unable to create engine";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
        exit(-1);
    }
    std::cout << "End building engine..." << std::endl;

}

void Net::BuildEngine() 
{
    std::cout << "Begin building engine..." << std::endl;
    engine_ = builder_->buildEngineWithConfig(*network_, *config_);
    std::cout << "Begin building engine..." << std::endl;
    if (!engine_){
        std::string error_message ="Unable to create engine";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
        exit(-1);
    }
    std::cout << "End building engine..." << std::endl;
}

void Net::SaveEngine(const std::string &fileName)
{
    if(engine_)
    {
        nvinfer1::IHostMemory* data = engine_->serialize();
        std::ofstream file;
        file.open(fileName,std::ios::binary | std::ios::out);
        if(!file.is_open())
        {
            std::cout << "read create engine file" << fileName <<" failed" << std::endl;
            return;
        }
        file.write((const char*)data->data(), data->size());
        file.close();
        data->destroy();
    }
}