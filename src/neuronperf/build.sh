#!/bin/bash

set -ex

python3 -m pytest -vv \
    --verbose \
    --ignore=build/private \
    --cov=neuronperf \
    --cov-report term-missing \
    --cov-report html:build/brazil-documentation/coverage \
    --cov-report xml:build/brazil-documentation/coverage/coverage.xml \
    --color=yes \
    -x \
    test \
    -m "sanity or slow"

python3 setup.py bdist_wheel --dist-dir build/pip/public/neuronperf
