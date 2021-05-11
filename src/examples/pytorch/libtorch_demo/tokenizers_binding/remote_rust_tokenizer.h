#ifndef __REMOTE_RUST_TOKENIZER_H__
#define __REMOTE_RUST_TOKENIZER_H__

#include <cstdint>

extern "C" {
    extern void remote_rust_encode(const char *input_arr, uint32_t* output_arr, uint32_t output_arr_len);
}

#endif // __REMOTE_RUST_TOKENIZER_H__
