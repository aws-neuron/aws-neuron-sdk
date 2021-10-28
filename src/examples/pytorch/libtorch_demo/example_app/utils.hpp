#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <torch/script.h>

std::string get_visible_cores_str(size_t num_neuron_cores, size_t cores_per_model);
std::string get_uuid();
torch::jit::script::Module get_model(const std::string& filename);

#endif // __UTILS_HPP__
