import torch
import torch.neuron
import os
import sys
import pandas as pd
import csv
import math
from collections import Counter

import numpy as np

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
        self.start_times = []
        self.batch_size = batch_size
        self.num_cores = num_cores

    def add_result(self, correct_count, inference_count, latency_array, end_times, start_times):
        self.correct_count += correct_count

        self.inference_count += inference_count
        self.latency_array.extend(latency_array)
        self.end_times.extend(end_times)
        self.start_times.extend(start_times)

    def report(self, f, window_size=1):
        assert(len(self.latency_array) != 0)
        p50_latency = np.percentile(self.latency_array, 50)
        p90_latency = np.percentile(self.latency_array, 90)
        p95_latency = np.percentile(self.latency_array, 95)
        p99_latency = np.percentile(self.latency_array, 99)
        p100_latency = np.percentile(self.latency_array, 100)


        def get_bucket(start, end):
            bucketed_start = math.floor(start / window_size) * window_size
            bucketed_end = math.ceil(end / window_size) * window_size
            # The check is to make sure that we ignore timestamps that are larger than the window size
            if bucketed_end - bucketed_start == window_size:
                return bucketed_start
            else:
                return None
            
        # Divide the timestamps into different buckets
        bucketed_timestamps = [get_bucket(start, end)
                            for start, end in zip(self.start_times, self.end_times)]
        # Count the values in each bucket
        counted_buckets = Counter(
            item for item in bucketed_timestamps if item is not None)
        # Normalize each bucket
        bucket_throughputs = [(key, value / window_size)
                            for key, value in sorted(counted_buckets.items())]
        
        busy_throughputs = [value for _, value in bucket_throughputs]
        max_throughput = max(busy_throughputs) * self.batch_size
        avg_throughput = sum(busy_throughputs) * self.batch_size / len(busy_throughputs)
        
        f.write("\n")
        f.write(
            "Maximum throughput = {} sentences/sec\n".format(int(max_throughput)))
        f.write("Average throughput = {} sentences/sec\n".format(int(avg_throughput)))

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
