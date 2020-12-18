import torch
import torch.neuron
import os
import sys
import pandas as pd

import shutil
import boto3
import botocore

from urllib.parse import urlparse
from urllib.parse import urlsplit

from transformers import BertTokenizer
from concurrent import futures

import time
import datetime
from datetime import date
import csv
import boto3
import botocore

import numpy as np
import multiprocessing


class BertTestDataset(torch.utils.data.Dataset):
    """Bert test dataset."""

    def __init__(self, tsv_file, tokenizer, max_length=128, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            tokenizer (callable = hugging face tokenizer):  Takes a string and encodes to standard input tensor set
            max_length (int): Maximum length that all input tensors will be padded to
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(tsv_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = list(reader)

        lines.pop(0)

        self.sentence_frame = pd.DataFrame(lines)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.sentence_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s1_raw = self.sentence_frame.iloc[idx, 3]
        if isinstance(s1_raw, bytes):
            s1_raw = s1_raw.decode("utf-8", "ignore")
        s2_raw = self.sentence_frame.iloc[idx, 4]
        if isinstance(s2_raw, bytes):
            s1_raw = s1_raw.decode("utf-8", "ignore")

        quality = self.sentence_frame.iloc[idx, 0]

        encoded = self.tokenizer.encode_plus(s1_raw, s2_raw, add_special_tokens=True,
                                             return_tensors='pt', max_length=self.max_length, 
                                             padding='max_length', truncation=True)

        sample = {'encoded': encoded, 'quality': quality}

        if self.transform:
            sample = self.transform(sample)

        return sample


class BertResults():

    def __init__(self, batch_size, num_cores=1):
        self.correct_count = 0
        self.inference_count = 0
        self.latency_array = []
        self.end_times = []
        self.total_latency_parallel = 0.0
        self.batch_size = batch_size
        self.num_cores = num_cores

    def add_result(self, correct_count, inference_count, latency_array, end_times, total_latency):
        self.correct_count += correct_count

        self.inference_count += inference_count
        self.latency_array.extend(latency_array)
        self.end_times.extend(end_times)
        self.total_latency_parallel += total_latency

    def report(self, f, bins=10):
        assert(len(self.latency_array) != 0)
        p50_latency = np.percentile(self.latency_array, 50)
        p90_latency = np.percentile(self.latency_array, 90)
        p95_latency = np.percentile(self.latency_array, 95)
        p99_latency = np.percentile(self.latency_array, 99)
        p100_latency = np.percentile(self.latency_array, 100)

        if self.total_latency_parallel == 0.0:
            self.total_latency_parallel = 1.0

        # Take all of the end time-stamps and construct a time binned histogram
        hist, bin_edges = np.histogram(self.end_times, bins=bins)

        overall_throughput = self.inference_count / \
            float(self.total_latency_parallel)

        f.write("\n")
        f.write("Histogram throughput (UTC times):\n")
        f.write("===\n")
        max_throughput = 0.0
        for i in range(len(hist)):
            delta = bin_edges[i+1] - bin_edges[i]
            # Each datestamp is batch size inferences
            throughput = self.batch_size * self.num_cores * hist[i] / delta
            if throughput > max_throughput:
                max_throughput = throughput
            st1 = datetime.datetime.fromtimestamp(
                bin_edges[i]).strftime('%H:%M:%S.%f')[:-3]
            st2 = datetime.datetime.fromtimestamp(
                bin_edges[i+1]).strftime('%H:%M:%S.%f')[:-3]
            f.write("{} - {} => {} sentences/sec\n".format(st1, st2, int(throughput)))
        f.write("\n")
        f.write(
            "Maximum throughput (histogram) = {} sentences/sec\n".format(int(max_throughput)))
        f.write("Overall throughput (aggregate stats * parallel) = {} sentences/sec\n".format(int(overall_throughput)))

        f.write("\n")
        f.write("Latency Percentiles:\n")
        f.write("===\n")
        f.write("P50  = {} milliseconds\n".format(int(1000*p50_latency)))
        f.write("P90  = {} milliseconds\n".format(int(1000*p90_latency)))
        f.write("P95  = {} milliseconds\n".format(int(1000*p95_latency)))
        f.write("P99  = {} milliseconds\n".format(int(1000*p99_latency)))
        f.write("P100 = {} milliseconds\n".format(int(1000*p100_latency)))
        f.write("\n")
        f.write("Accuracy:\n")
        f.write("===\n")
        if self.inference_count == 0:
            self.inference_count = 1
        accuracy = float(self.correct_count) / float(self.inference_count)
        f.write("Accuracy = {}% \n".format(round(100*accuracy, 2)))
        f.write("\n")
        f.write("Sanity test:\n")
        f.write("===\n")
        f.write("Processed - num batches {}\n".format(len(self.latency_array)))
        f.write("          - batch size {}\n".format(self.batch_size))
        f.write("          - num cores {}\n".format(self.num_cores))
