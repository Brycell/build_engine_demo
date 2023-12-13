/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENTROPY_CALIBRATOR_H
#define ENTROPY_CALIBRATOR_H

#include "BatchStream.h"
#include <iterator>
#include "NvInfer.h"

//! \class EntropyCalibratorImpl
//!
//! \brief Implements common functionality for Entropy calibrators.
//!
template <typename TBatchStream>
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(TBatchStream const& stream, int firstBatch, std::string const& networkName, bool readCache = true)
        : mStream{stream}
        , mCalibrationTableName("CalibrationTable_" + networkName)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = volume(dims);
        std::cout<<"mInputCount:"<<mInputCount<<std::endl;
        cudaMalloc(&mDeviceLInput, mInputCount * sizeof(float));
        // cudaMalloc(&mDeviceRInput, mInputCount * sizeof(float));
        mStream.reset(firstBatch);
    }

    virtual ~EntropyCalibratorImpl()
    {
        cudaFree(mDeviceLInput);
        // cudaFree(mDeviceRInput);
    }

    int getBatchSize() const noexcept
    {
        std::cout<<"getBatchSize"<<std::endl;
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
    {
        std::cout<<"getBatch"<<std::endl;
        if (!mStream.next())
        {
            return false;
        }
        cudaMemcpy(mDeviceLInput, mStream.getLeftBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice);
        std::cout<<"names[0]:"<<names[0]<<std::endl;
        bindings[0] = mDeviceLInput;

        // cudaMemcpy(mDeviceRInput, mStream.getRightBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice);
        // std::cout<<"names[1]:"<<names[1]<<std::endl;
        // bindings[1] = mDeviceRInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept
    {
        std::cout<<"readCalibrationCache"<<std::endl;
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept
    {
        std::cout<<"writeCalibrationCache"<<std::endl;
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    TBatchStream mStream;
    size_t mInputCount;
    std::string mCalibrationTableName;
    bool mReadCache{true};
    void* mDeviceLInput{nullptr};
    // void* mDeviceRInput{nullptr};
    std::vector<char> mCalibrationCache;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
template <typename TBatchStream>
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(TBatchStream const& stream, int32_t firstBatch, const char* networkName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, readCache)
    {
    }

    int getBatchSize() const noexcept override
    {
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};

#endif // ENTROPY_CALIBRATOR_H
