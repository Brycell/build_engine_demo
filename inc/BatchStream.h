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
#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "filesystem/directory.h"

static int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getLeftBatch() = 0;
    // virtual float* getRightBatch() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};

class MNISTBatchStream : public IBatchStream
{
public:
    MNISTBatchStream(int batchSize, int maxBatches, const std::string& dir_path)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{3, {3, 768, 1280}} //!< We already know the dimensions of MNIST images.
    {
        readDataFile(dir_path);
    }
    
    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount >= mMaxBatches)
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getLeftBatch() override
    {
        std::cout<<"mBatchCount:"<<mBatchCount<<std::endl;
        return mLeftData.data() + ((mBatchCount-1) * mBatchSize * volume(mDims));
    }
    // float* getRightBatch() override
    // {
    //     std::cout<<"mBatchCount:"<<mBatchCount<<std::endl;
    //     return mRightData.data() + ((mBatchCount-1) * mBatchSize * volume(mDims));
    // }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return nvinfer1::Dims{4, {mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]}};
    }

private:
    void readDataFile(const std::string& dir_path)
    {
        filesystem::directory left_dir(dir_path + "/left_images");
        std::vector<std::string> left_list;
        for(const auto &p : left_dir) {
            if(p.extension() == "png") {
                left_list.push_back(dir_path + "/left_images/" + p.filename());
           }
        }
        std::sort(left_list.begin(), left_list.end());

        // filesystem::directory right_dir(dir_path + "/right_images");
        // std::vector<std::string> right_list;
        // for(const auto &p : right_dir) {
        //     if(p.extension() == "png") {
        //         right_list.push_back(dir_path + "/right_images/" + p.filename());
        //     }
        // }
        // std::sort(right_list.begin(), right_list.end());


        int channel = mDims.d[0];
        int height = mDims.d[1];
        int width = mDims.d[2];
        int image_size = height * width;
        int numElements = mMaxBatches * channel * image_size;
        std::cout<<"numElements:"<<numElements<<std::endl;
        mLeftData.resize(numElements);
        // mRightData.resize(numElements);
        for(int i = 0; i < mMaxBatches; i++) {
            std::cout<<left_list[i]<<std::endl;
            cv::Mat left = cv::imread(left_list[i])(cv::Rect(1184+128, 202+640, 1280, 768)).clone();
            // cv::Mat right = cv::imread(right_list[i])(cv::Rect(1184+128, 202, 1216, 1408)).clone();
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int offset =  h * width + w;
                    for(int c = 0; c < channel; c++) {
                        int src_index = offset * channel + c;
                        int dst_index = image_size * c + offset;
                        mLeftData[dst_index] = *(left.data + src_index) * 1.f - 128.f;
                        // mRightData[dst_index] = *(right.data + src_index) * 1.f - 128.f;
                    }
               }
            } 
            std::cout<<"read image:"<<i<<std::endl;
        }
    }

    int mBatchSize{0};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    nvinfer1::Dims mDims{};
    std::vector<float> mLeftData{};
    // std::vector<float> mRightData{};
};
#endif