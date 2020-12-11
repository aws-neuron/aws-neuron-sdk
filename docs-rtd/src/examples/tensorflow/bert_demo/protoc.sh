#!/bin/bash
# coding=utf-8

""" Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
    Program to gather information from a system
"""

python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. mrpc.proto
