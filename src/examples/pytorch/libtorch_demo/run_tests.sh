#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "usage: ./run_tests.sh model_filename.pt"
    exit 1
fi

echo -e "\nRunning tokenization sanity checks.\n"
pushd tokenizers_binding 2>&1 >/dev/null
chmod +x run_python.sh run.sh
(./run_python.sh && ./run.sh) || { echo "Sanity checks failed."; exit 2; }
popd 2>&1 >/dev/null
echo -e "\nTokenization sanity checks passed."

echo -e "Running end-to-end sanity check.\n"
(./example-app $1 --sanity) || { echo "Sanity check failed."; exit 3; }
echo -e "\nSanity check passed.\n"