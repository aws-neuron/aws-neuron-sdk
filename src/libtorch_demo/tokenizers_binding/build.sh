#!/bin/bash

# clean old artifacts
rm tokenizer_test 2>&1 >/dev/null
rm -rf lib 2>&1 >/dev/null

# build shared library
if [ $# -eq 0 ]; then
    pushd ../tokenizers/tokenizers
    echo "Building release test..."
    cargo build --release
    popd
    cp -r ../tokenizers/tokenizers/target/release lib
    g++ -O3 -o tokenizer_test tokenizer_test.cpp -L./lib -ltokenizers
else
    pushd ../tokenizers/tokenizers
    echo "Building debug test..."
    cargo build
    popd
    cp -r ../tokenizers/tokenizers/target/debug lib
    g++ -O0 -o tokenizer_test tokenizer_test.cpp -L./lib -ltokenizers
fi

if [ ! -e "tokenizer.json" ]; then
    wget https://huggingface.co/bert-base-cased-finetuned-mrpc/raw/main/tokenizer.json
fi
