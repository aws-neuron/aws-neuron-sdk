#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "utils.hpp"
#include "../tokenizers_binding/remote_rust_tokenizer.h"

typedef std::vector<std::vector<long>> Input;

namespace
{
    // some hardcoded parameters that could be read from a config file
    const size_t seq_len = 128;
    const size_t batch_size = 6;
    const size_t num_neuron_cores = 4;
    const size_t cores_per_model = 1;
    const size_t num_runs_per_neuron_core = 1000;

    // these token ids are particular to a vocabulary, could be parsed from vocab file
    const long start_token = 101;
    const long end_token = 102;
}

// construct a single input: input_ids, attention_mask, and token_type_ids from two input sentences
Input get_input(const std::string& sentence_1, const std::string& sentence_2)
{
    // ensure the concatenated sentences + separator tokens do not exceed the compiled sequence length
    assert(sentence_1.size() + sentence_2.size() + 3 <= seq_len);

    // tokenize the input sentence using the HuggingFace Tokenizers library
    std::vector<long> input_ids(seq_len, 0);
    input_ids[0] = start_token;
    size_t pos = 1; // current write position in input_ids

    // tokenize sentence_1 and copy to output buffer
    std::vector<uint32_t> buffer(seq_len, 0);
    remote_rust_encode(sentence_1.c_str(), buffer.data(), buffer.size());
    for (size_t i = 0; i < seq_len && buffer[i]; i++, pos++) {
        input_ids[pos] = buffer[i];
    }

    // mark end of sentence_1
    input_ids[pos++] = end_token;
    const size_t sentence_2_start = pos;

    // tokenize sentence_2 and copy to output buffer
    std::fill(buffer.begin(), buffer.end(), 0);
    remote_rust_encode(sentence_2.c_str(), buffer.data(), buffer.size());
    for (size_t i = 0; i < seq_len && buffer[i]; i++, pos++) {
        input_ids[pos] = buffer[i];
    }

    // mark end of sentence_2
    input_ids[pos++] = end_token;

    // construct attention mask
    std::vector<long> attention_mask(seq_len, 0);
    for (size_t i = 0; i < seq_len; ++i) attention_mask[i] = input_ids[i] ? 1 : 0;

    // token type ids are 0s for sentence_1 (incl. separators), 1s for sentence_2
    std::vector<long> token_type_ids(seq_len, 0);
    for (size_t i = sentence_2_start; i < seq_len; i++) {
        if (!attention_mask[i]) break;
        token_type_ids[i] = 1;
    }

    return {input_ids, attention_mask, token_type_ids};
}

// reshape a vector of inputs into a proper batch
std::vector<torch::jit::IValue> get_batch(const std::vector<Input>& inputs)
{
    // must be given a full batch
    assert(inputs.size() == batch_size);

    torch::Tensor input_ids_tensor = torch::zeros({batch_size, seq_len}, at::kLong);
    torch::Tensor attention_mask_tensor = torch::zeros({batch_size, seq_len}, at::kLong);
    torch::Tensor token_type_ids_tensor = torch::zeros({batch_size, seq_len}, at::kLong);

    const auto opts = torch::TensorOptions().dtype(torch::kLong);
    for (size_t i = 0; i < batch_size; i++) {
        input_ids_tensor.slice(0, i, i+1) = torch::from_blob((void*)inputs[i][0].data(), {seq_len}, opts);
        attention_mask_tensor.slice(0, i, i+1) = torch::from_blob((void*)inputs[i][1].data(), {seq_len}, opts);
        token_type_ids_tensor.slice(0, i, i+1) = torch::from_blob((void*)inputs[i][2].data(), {seq_len}, opts);
    }

    return {input_ids_tensor, attention_mask_tensor, token_type_ids_tensor};
}

int sanity_check(const std::string& model_filename)
{
    // load the model
    auto model = get_model(model_filename);

    // construct some example inputs
    const std::string sentence_1 = "The company HuggingFace is based in New York City";
    const std::string sentence_2 = "Apples are especially bad for your health";
    const std::string sentence_3 = "HuggingFace's headquarters are situated in Manhattan";
    const auto paraphrase = get_input(sentence_1, sentence_3);
    const auto not_paraphrase = get_input(sentence_1, sentence_2);

    // batch the inputs 50/50 positive/negative
    std::vector<Input> inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        if (i < batch_size / 2) {
            inputs[i] = paraphrase;
        } else {
            inputs[i] = not_paraphrase;
        }
    }
    const auto batch = get_batch(inputs);

    // forward pass
    const auto output = model.forward(batch);

    // interpret output
    const auto output_tensor = output.toTuple()->elements()[0].toTensor();
    const auto paraphrase_probabilities = torch::softmax(output_tensor[0], 0);
    const auto not_paraphrase_probabilities = torch::softmax(output_tensor[batch_size-1], 0);
    const auto paraphrase_0 = std::round(paraphrase_probabilities[0].item<double>() * 100);
    const auto paraphrase_1 = std::round(paraphrase_probabilities[1].item<double>() * 100);
    const auto not_paraphrase_0 = std::round(not_paraphrase_probabilities[0].item<double>() * 100);
    const auto not_paraphrase_1 = std::round(not_paraphrase_probabilities[1].item<double>() * 100);

    std::cout << sentence_1 << std::endl << sentence_3 << std::endl;
    std::cout << "not paraphrase: " << paraphrase_0 << "%" << std::endl;
    std::cout << "paraphrase: " << paraphrase_1 << "%" << std::endl;
    if (paraphrase_0 >= paraphrase_1) return -1;

    std::cout << std::endl;

    std::cout << sentence_1 << std::endl << sentence_2 << std::endl;
    std::cout << "not paraphrase: " << not_paraphrase_0 << "%" << std::endl;
    std::cout << "paraphrase: " << not_paraphrase_1 << "%" << std::endl;
    if (not_paraphrase_0 <= not_paraphrase_1) return -2;

    return 0;
}

void benchmark(const std::string& model_filename, const std::vector<torch::jit::IValue>& batch,
               std::condition_variable& warmup_cv, std::atomic_size_t& warmup_count,
               std::condition_variable& ready_cv)
{
    // load model and warmup
    auto model = get_model(model_filename);
    model.forward(batch);
    std::cout << "." << std::flush;
    --warmup_count;
    warmup_cv.notify_one();

    // wait for ready signal
    std::mutex ready_mutex;
    std::unique_lock<std::mutex> lk(ready_mutex);
    ready_cv.wait(lk);

    // benchmark
    for (size_t i = 0; i < num_runs_per_neuron_core; i++) {
        if (i == num_runs_per_neuron_core/2) std::cout << "." << std::flush;
        model.forward(batch);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./example_app neuron_traced_model.pt [--sanity]" << std::endl;
        return -1;
    }

    // let runtime know we want M models / core for N cores (e.g. "1,1,1,1")
    setenv("NEURON_RT_VISIBLE_CORES", get_visible_cores_str(num_neuron_cores, cores_per_model).c_str(), true);

    if (argc >= 3 && std::string("--sanity") == argv[2]) {
        return sanity_check(argv[1]);
    }

    /*************************************************************************/
    // prepare inputs, prepare models, and perform warmup inference

    std::cout << "Getting ready" << std::flush;

    const auto input = get_input("This sentence is for benchmarking.", "For benchmarking, use this sentence.");
    const auto batch = get_batch(std::vector<Input>(batch_size, input));

    std::condition_variable warmup_cv, ready_cv;
    std::atomic_size_t warmup_count(num_neuron_cores);
    std::vector<std::thread> threads(num_neuron_cores);
    for (size_t i = 0; i < threads.size(); i++) {
        threads[i] = std::move(std::thread(benchmark, argv[1], batch, std::ref(warmup_cv),
                                std::ref(warmup_count), std::ref(ready_cv)));
    }

    // wait for warmup to complete
    auto is_warmup_complete = [](std::atomic_size_t& warmup_count) { return warmup_count.load() == 0; };
    std::mutex warmup_mutex;
    std::unique_lock<std::mutex> lk(warmup_mutex);
    warmup_cv.wait(lk, std::bind(is_warmup_complete, std::ref(warmup_count)));
    std::cout << std::endl;

    /*************************************************************************/
    // begin timed benchmarking

    std::cout << "Benchmarking" << std::flush;

    // signal workers to begin benchmarking and wait for completion
    const auto start_time = std::chrono::high_resolution_clock::now();
    ready_cv.notify_all();
    for (auto& thread : threads) thread.join();
    const auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;

    // report statistics
    const float elapsed = (end_time - start_time) / std::chrono::seconds(1);
    const size_t num_inferences = num_neuron_cores * num_runs_per_neuron_core;
    const float throughput = (float)(num_inferences * batch_size) / elapsed;
    std::cout << "Completed " << num_inferences << " operations in " << elapsed << " seconds => " << throughput << " pairs / second" << std::endl;

    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Summary information:" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Batch size = " << batch_size << std::endl;
    std::cout << "Num neuron cores = " << num_neuron_cores << std::endl;
    std::cout << "Num runs per neruon core = " << num_runs_per_neuron_core << std::endl;

    return 0;
}
