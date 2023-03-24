#!/bin/bash

echo "Clean up constructed files"
rm -rf bert_neuron_b6.pt example-app tokenizers venv/ libtorch/ tokenizers_binding/lib/ tokenizers_binding/venv all_metrics.csv venv
