#!/bin/bash

if [ ! -e "venv" ]; then
    python3 -m venv venv
    . venv/bin/activate
    pip install -U pip
    pip install tqdm==4.56.0 transformers==4.2.2
    deactivate
fi