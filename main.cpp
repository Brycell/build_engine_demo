#include <argparse.h>
#include <string>
#include <iostream>
#include "net.h"

int main(int argc, const char** argv)
{
    optparse::OptionParser parser;
    parser.add_option("-i", "--input-onnx-file").dest("onnxFile")
            .help("the path of onnx file");
    parser.add_option("-o", "--output-engine-file").dest("outputFile")
            .help("the path of engine file");
    parser.add_option("-m", "--mode").dest("mode").set_default<int>(0)
            .help("run-mode, type int");
    parser.add_option("-c", "--calib").dest("calibFile").help("calibFile, type str");
    optparse::Values options = parser.parse_args(argc, argv);
    if(options["onnxFile"].size() == 0){
        std::cout << "no file input" << std::endl;
        exit(-1);
    }
    RUN_MODE mode = RUN_MODE::FLOAT32;
    if(options["mode"] == "0" ) mode = RUN_MODE::FLOAT32;
    if(options["mode"] == "1" ) mode = RUN_MODE::FLOAT16;
    if(options["mode"] == "2" ) mode = RUN_MODE::INT8;

    Net net;
    std::string dir_path = "/algdata/zkhy/input8M/2023_11_28_13_45/images_004/";
    net.InitEngine(options["onnxFile"], dir_path, mode);
//     net.BuildEngine();
    net.SaveEngine(options["outputFile"]);

    std::cout << "save  " << options["outputFile"] <<std::endl;
}