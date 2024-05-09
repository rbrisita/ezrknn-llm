// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Modified by:
 * @Pelochus
 * @rbrisita
*/

#include <csignal>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "rkllm.h"

#define SYSTEM_PROMPT(s) "<|im_start|>system " + s + " <|im_end|>"
#define USER_PROMPT(s) "<|im_start|>user " + s + " <|im_end|>"
#define ASSISTANT_PROMPT_PREFIX "<|im_start|>assistant"

using namespace std;

LLMHandle llmHandle = nullptr;

bool quiet;                     // -q
std::string modelPath;          // -m
std::string systemPrompt = "You are a helpful assistant."; // -s

int32_t num_npu_core = 1;       // -n
int32_t max_context_len = 512;  // -c
int32_t max_new_tokens = -1;    // -t

int32_t top_k = 40;             // -K
float top_p = 0.9;              // -P
float temperature = 0.8;        // -T

float repeat_penalty = 1.1;     // -r
float frequency_penalty = 0;    // -f
float presence_penalty = 0;     // -p

int32_t mirostat = 0;           // -v
float mirostat_tau = 5;         // -e
float mirostat_eta = 0.1;       // -l


void usage() {
    std::cout <<
        "-q --quiet:                               Quiet mode\n"
        "-m --model <model file>:                  RKLLM model file to run\n"
        "-s --system <text>:                       System prompt\n"
        "-n --num <int>:                           Number of NPU cores to use (1)\n"
        "-c --context <int>:                       Set context window (512)\n"
        "-t --tokens <int>:                        Set max tokens generated (-1)\n"
        "\nCreativity\n"
        "-K --top_k <int>:                         Set sampling (reduced vocabulary) (40)\n"
        "-P --top_p <0 <= float <= 1>:             Set nucleus sampling (vocabulary size) (0.9)\n"
        "-T --temp <0 <= float <= 2>:              Set randomness level (0.8)\n"
        "\nPenalty\n"
        "-r --repeat_penalty <0 <= float <= 100>:  Reduce token reuse (1.1)\n"
        "-f --frequency_penalty <-2 <= float <= 2>:Adjust sampling of tokens (decrease likelihood to repeat the same line verbatim) (0)\n"
        "-p --presence_penalty <-2 <= float <= 2>: Adjust sampling of tokens (increase likelihood to talk about new topics) (0)\n"
        "\nPerplexity\n"
        "-v --mirostat <int>:                      Set version level (0)\n"
        "-e --mirostat_tau <float>:                Set target entropy (5)\n"
        "-l --mirostat_eta <0.05 <= float <= 0.2>: Set learning rate (0.1)\n"
        "\n\n"
        "-h --help:                                Show this help\n";
    exit(1);
}


void parseArgs(int argc, char **argv) {
    const char* const short_opts = "qm:s:n:c:t:K:P:T:r:f:p:v:e:l:h";
    const option long_opts[] = {
            {"quiet", no_argument, nullptr, 'q'},
            {"model", required_argument, nullptr, 'm'},
            {"system", required_argument, nullptr, 's'},
            {"num", required_argument, nullptr, 'n'},
            {"context", required_argument, nullptr, 'c'},
            {"tokens", required_argument, nullptr, 't'},
            {"top_k", required_argument, nullptr, 'K'},
            {"top_p", required_argument, nullptr, 'P'},
            {"temp", required_argument, nullptr, 'T'},
            {"repeat_penalty", required_argument, nullptr, 'r'},
            {"frequency_penalty", required_argument, nullptr, 'f'},
            {"presence_penalty", required_argument, nullptr, 'p'},
            {"mirostat", required_argument, nullptr, 'v'},
            {"mirostat_tau", required_argument, nullptr, 'e'},
            {"mirostat_eta", required_argument, nullptr, 'l'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'q':
            quiet = true;
            break;

        case 'm':
            modelPath = std::string(optarg);
            break;

        case 's':
            systemPrompt = std::string(optarg);
            break;

        case 'n':
            num_npu_core = std::stoi(optarg);
            break;

        case 'c':
            max_context_len = std::stoi(optarg);
            break;

        case 't':
            max_new_tokens = std::stoi(optarg);
            break;

        case 'K':
            top_k = std::stoi(optarg);
            break;

        case 'P':
            top_p = std::stof(optarg);
            break;

        case 'T':
            temperature = std::stof(optarg);
            break;

        case 'r':
            repeat_penalty = std::stof(optarg);
            break;

        case 'f':
            frequency_penalty = std::stof(optarg);
            break;

        case 'p':
            presence_penalty = std::stof(optarg);
            break;

        case 'v':
            mirostat = std::stoi(optarg);
            break;

        case 'e':
            mirostat_eta = std::stof(optarg);
            break;

        case 'l':
            mirostat_tau = std::stof(optarg);
            break;

        case 'h': // -h or --help
        case '?': // Unrecognized option
        default:
            usage();
            break;
        }
    }
}


const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string &s) {
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s) {
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string &s) {
    return rtrim(ltrim(s));
}


void exit_handler(int signal) {
    if (llmHandle != nullptr) {
        cout << "Catched exit signal. Exiting..." << endl;
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
        exit(signal);
    }
}


void callback(const char *text, void *userdata, LLMCallState state) {
    switch(state) {
        case LLM_RUN_NORMAL:
            printf("%s", text);
            break;
        case LLM_RUN_FINISH:
            printf("\n");
            break;
        case LLM_RUN_ERROR:
            printf("\\LLM run error\n");
            break;
        default:
            break;
    }
}

int main(int argc, char **argv) {
    parseArgs(argc, argv);

    // Trim and check
    modelPath = trim(modelPath);
    if (modelPath.empty()) {
        printf("Model file needed.\n");
        printf("Usage: %s <-m | --model <rkllm_model_path>>\n", argv[0]);
        usage();
        return -1;
    }

    std::cout << "RKLLM starting, please wait..." << endl;

    signal(SIGINT, exit_handler);

    RKLLMParam param = rkllm_createDefaultParam();
    param.modelPath = modelPath.c_str();
    param.target_platform = "rk3588";
    param.num_npu_core = num_npu_core;
    param.max_context_len = max_context_len;
    param.max_new_tokens = max_new_tokens;
    param.top_k = top_k;
    param.top_p = top_p;
    param.temperature = temperature;
    param.repeat_penalty = repeat_penalty;
    param.frequency_penalty = frequency_penalty;
    param.presence_penalty = presence_penalty;
    param.mirostat = mirostat;
    param.mirostat_tau = mirostat_tau;
    param.mirostat_eta = mirostat_eta;

    if (!quiet) {
        std::cout << "param.modelPath " << param.modelPath
        << "\nparam.target_platform " << param.target_platform
        << "\nparam.num_npu_core " << param.num_npu_core
        << "\nparam.max_context_len " << param.max_context_len
        << "\nparam.max_new_tokens " << param.max_new_tokens
        << "\nparam.top_k " << param.top_k
        << "\nparam.top_p " << param.top_p
        << "\nparam.temperature " << param.temperature
        << "\nparam.repeat_penalty " << param.repeat_penalty
        << "\nparam.frequency_penalty " << param.frequency_penalty
        << "\nparam.presence_penalty " << param.presence_penalty
        << "\nparam.mirostat " << param.mirostat
        << "\nparam.mirostat_tau " << param.mirostat_tau
        << "\nparam.mirostat_eta " << param.mirostat_eta
        << endl;
    }

    int result = rkllm_init(&llmHandle, param, callback);
    if (result != 0) {
        return result;
    }

    std::cout << "RKLLM init success!\n"
    << "Enter either 'exit' or 'quit' to terminate."
    << endl;
    sleep(0.1);

    systemPrompt = SYSTEM_PROMPT(systemPrompt);
    std::string text;
    std::string input_str;
    while (true) {
        printf(">> "); // alert caller for input
        fflush(stdout);
        std::getline(std::cin, input_str);

        if (input_str == "exit" || input_str == "quit") {
            break;
        }

        text = systemPrompt
        + USER_PROMPT(input_str)
        + ASSISTANT_PROMPT_PREFIX;
        rkllm_run(llmHandle, text.c_str(), NULL);
    }

    rkllm_destroy(llmHandle);

    return 0;
}
