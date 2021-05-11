#include <iostream>
#include <chrono> // timing
#include <cstring> // rust interface
#include <iomanip> // std::setprecision
#include <sstream> // parse args
#include <vector>

#include "remote_rust_tokenizer.h"

#define DEFAULT_NUM_TESTS 10000u

int main(int argc, char *argv[]) {
    // prepare some input to tokenize
    const uint32_t seq_len = 128;
    const std::vector<uint32_t> ground_truth = { 1409, 1917, 2947, 16193, 117, 1142, 3087, 1209, 1129, 22559, 2200, 1656, 155, 8954, 119 };
    const char *input_arr = "If everything goes smoothly, this text will be tokenized inside Rust.";
    uint32_t* output_arr = new uint32_t[seq_len];
    std::memset(output_arr, 0, sizeof(uint32_t) * seq_len);

    // call rust tokenizer
    remote_rust_encode(input_arr, output_arr, seq_len);

    // check output
    std::cout << "Sanity check ";
    for (auto i = 0; i < ground_truth.size(); ++i) {
        if (output_arr[i] != ground_truth[i]) {
            std::cerr << "failed at: " << i << ", " << output_arr[i] << " != " << ground_truth[i] << std::endl;
            return -1;
        }
    }
    std::cout << "passed." << std::endl;

    // run timed test
    uint32_t num_tests = DEFAULT_NUM_TESTS;
    if (argc >= 3 && !strcmp("--num_tests", argv[1])) {
        std::istringstream iss(argv[2]);
        iss >> num_tests;
    }

    const uint32_t ten_percent = uint32_t(0.1 * num_tests);
    std::cout << "Begin " << num_tests << " timed tests." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (auto test_num = 0; test_num < num_tests; ++test_num) {
        if (test_num % ten_percent == 0) {
            std::cout << "." << std::flush;
        }
        remote_rust_encode(input_arr, output_arr, seq_len);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    std::cout << std::endl << "End timed tests." << std::endl << "C++ took "
        << std::setprecision(3) << duration.count()
        << " seconds." <<  std::endl;

    return 0;
}
