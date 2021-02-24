#!/bin/bash

set -e

. venv/bin/activate
python tokenizer_test.py
deactivate
