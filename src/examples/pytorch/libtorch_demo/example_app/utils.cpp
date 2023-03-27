#include "utils.hpp"
#include "../tokenizers_binding/remote_rust_tokenizer.h"

#include <random>
#include <sstream>

#include <torch/csrc/jit/passes/inliner.h>
#include <ATen/ATen.h>

std::string get_visible_cores_str(size_t num_neuron_cores, size_t cores_per_model)
{
    std::ostringstream oss;
    oss << "0-" << ((num_neuron_cores * cores_per_model) - 1);
    return oss.str();
}

std::string get_uuid()
{
    // xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
    // M = version = 4, (4 bits, 0100 = 0x4)
    // N = variant = 1, (2 bits, 10XX = 0x{8, 9, A, B})

    static const char *chars = "0123456789abcdef";
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_int_distribution<> dist(0, 15);

    std::stringstream ss;
    for (size_t i = 0; i < 37; i++) {
        const int index = dist(mt);
        ss << chars[index];
    }

    // variant bits are 10XX
    std::stringstream variant_ss;
    size_t variant;
    variant_ss << std::hex << chars[dist(mt)];
    variant_ss >> variant;
    variant = 0x8 | (0x3 & variant);

    ss.seekp(9); ss << "-";
    ss.seekp(14); ss << "-4";
    ss.seekp(19); ss << "-" << std::hex << variant;
    ss.seekp(24); ss << "-";
    return ss.str();
}

torch::jit::script::Module get_model(const std::string& filename)
{
    torch::jit::script::Module model = torch::jit::load(filename);

    // If you're using a model traced with torch-neuron >= 1.8, 
    // the section below is no longer necessary. It was a workaround 
    // for a runtime issue when loading identical copies of a model.

    // This is redundant in the new flow, but left to provide future 
    // pointer on torchscript graph manipulation if needed

    // this next section adds a unique uuid to the graph, so that the neuron runtime
    // will load the graph multiple times instead of reusing a previously loaded copy

    /*
    auto fwd = model.get_method("forward");
    auto& fn = static_cast<torch::jit::GraphFunction&>(fwd.function());
    auto graph = fn.graph();

    torch::jit::Inline(*graph);
    for (auto node : graph->nodes()) {
        if (std::string(node->kind().toQualString()).rfind("neuron::forward") == 0) {
            auto uuid_input_tensor = node->inputs()[1];
            if (std::string(uuid_input_tensor->node()->kind().toQualString()).rfind("prim::Constant") == 0) {
                // we clone the tensor to retain ownership of "the blob" after it goes out of scope
                const std::string uuid = get_uuid();
                torch::Tensor t = torch::from_blob((void*)uuid.c_str(), {36}, torch::kUInt8).clone();

                // if we don't move the insertion point so that the copy of the constant appears after the operator,
                // the inference will crash
                graph->setInsertPoint(node);
                torch::jit::Value *val = graph->insertConstant(t);
                node->replaceInputWith(uuid_input_tensor, val);

                // ensure a valid graph
                graph->lint();
            }
        }
    }
    */

    return model;
}
