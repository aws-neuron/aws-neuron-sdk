import logging
import math
import os
import tempfile
import time
import unittest
from contextlib import contextmanager
from uuid import uuid4
import torch
from transformers_neuronx import utils, dtypes


class SamplingBenchmarkTest(unittest.TestCase):
    @contextmanager
    def apply_latest_compiler_options(self):
        compiler_options = " --tensorizer-options=' --enable-tritium-loopfusion' --model-type=transformer-inference --internal-mm-reorder-opt"
        with self.updated_env_context(NEURON_CC_FLAGS=compiler_options) as updated_env:
            try:
                yield
            finally:
                pass

    @contextmanager
    def updated_env_context(self, **update):
        """
        Temporarily updates the os.environ dictionary in-place.

        :param update: Dictionary of environment variables and values to add/update.
        """
        env = os.environ
        update = update or {}
        stomped = set(update.keys()) & set(env.keys())
        update_after = {k: env[k] for k in stomped}
        remove_after = frozenset(k for k in update if k not in env)

        try:
            env.update(update)
            yield
        finally:
            env.update(update_after)
            [env.pop(k) for k in remove_after]

    @contextmanager
    def model_dir_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            *_, test_name = self.id().split('.')
            model_dir = os.path.join(tmpdir, f'model-{test_name}')
            yield model_dir

    def short_batched_inputs(self, tokenizer, batch_size):
        text = "Hello, I'm a language model,"
        return tokenizer([text for _ in range(batch_size)], return_tensors='pt')

    def publish_heartbeat_metrics(self, test_name, report_dict, test_parameters):
        try:
            from heartbeat import Heartbeat
        except ImportError:
            print('Heartbeat Installation Not Found. Skipping Posting Metrics')
            return

        # Upload metrics to HB
        heartbeat = Heartbeat()
        logging.debug(f"Posting metrics to Heartbeat 2.0 for test: {test_name}")

        test_data = {
            "Model": test_parameters["model_name"],
            "Data type": str(test_parameters["dtype"]),
            "Hidden layers": test_parameters["num_hidden_layers"],
            "Batch Size": test_parameters["batch_size"],
            "Sequence length": test_parameters["sequence_length"],
            "Hidden size": test_parameters["hidden_size"],
            "Vocab size": test_parameters["vocab_size"],
            "TP degree": test_parameters["tp_degree"]
        }

        units = {
            "Latency P0": "ms",
            "Latency P50": "ms",
            "Latency P90": "ms",
            "Latency P95": "ms",
            "Latency P99": "ms",
            "Latency P100": "ms",
            "Average throughput": "inf/s",
            "Peak throughput": "inf/s",
            "HBM bandwidth": "%",
        }

        test_metrics = []
        for key, value in report_dict.items():
            test_metrics.extend([
                {
                    "MetricName": key,
                    "MeasuredValue": value,
                    "Units": units[key],
                }
            ])

        dimensions = {"ExecutionId": str(uuid4())}
        heartbeat.post_metrics(
            identifier=test_name,
            parameters=test_data,
            metrics=test_metrics,
            dimensions=dimensions,
        )

    def benchmark(self, model, model_name, inputs, sequence_length, warmup,
                  batch_size, hidden_size, ffn_dim, num_hidden_layers,
                  vocab_size, amp, tp_degree, n_runs=5, **generation_kwargs):
        *_, test_name = self.id().split('.')
        dtype, dtype_layers, _ = utils.parse_amp(amp)
        dtype = dtypes.to_torch_dtype(dtype)
        test_parameters = dict(
            model_name=model_name, dtype=dtype, sequence_length=sequence_length,
            batch_size=batch_size, hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers, vocab_size=vocab_size, tp_degree=tp_degree,
        )
        test_parameters.update(generation_kwargs)
        generated_sequence = []
        report_dict = {}
        try:
            generated_sequence, report_dict = self.benchmark_impl(
                model, inputs, sequence_length, warmup, batch_size, hidden_size,
                ffn_dim, num_hidden_layers, vocab_size, amp, tp_degree, n_runs, **generation_kwargs
            )
        finally:
            self.publish_heartbeat_metrics(test_name, report_dict, test_parameters)
        return generated_sequence

    def benchmark_impl(self, model, inputs, sequence_length, warmup,
                       batch_size, hidden_size, ffn_dim, num_hidden_layers,
                       vocab_size, amp, tp_degree, n_runs, **generation_kwargs):
        # NOTE: n_runs should be 1 at least
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        _, start_ids = attention_mask.max(axis=1)
        if (start_ids == 0).all():
            start_ids = None
        elapsed_list = []
        latency_collector_all = LatencyCollector()
        model.to_neuron()
        with torch.inference_mode():
            if warmup:
                if len(generation_kwargs) == 0:
                    generated_sequence = model.sample(input_ids, sequence_length=sequence_length, start_ids=start_ids)
                else:
                    model.reset()
                    generated_sequence = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)

            model.register_forward_pre_hook(latency_collector_all.pre_hook)
            model.register_forward_hook(latency_collector_all.hook)
            for _ in range(n_runs):
                start = time.time()
                if len(generation_kwargs) == 0:
                    generated_sequence = model.sample(input_ids, sequence_length=sequence_length, start_ids=start_ids)
                else:
                    model.reset()
                    generated_sequence = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
                elapsed = time.time() - start
                elapsed_list.append(elapsed)

        num_prompt_tokens = input_ids.shape[-1]
        _, num_tokens = generated_sequence.shape
        num_new_tokens = num_tokens - num_prompt_tokens
        dtype, dtype_layers, _ = utils.parse_amp(amp)
        if dtype_layers is None:
            dtype_layers = dtype
        num_bytes_per_parameter_map = {'u8': 1, 's8': 1, 'bf16': 2, 'f16': 2, 'f32': 4}
        num_bytes_per_parameter = num_bytes_per_parameter_map[dtype]
        num_bytes_per_parameter_layers = num_bytes_per_parameter_map[dtype_layers]
        # ignore input embeddings (on cpu); ignore biases and LayerNorm parameters (too small)
        parameters_bytes = hidden_size * vocab_size * num_bytes_per_parameter
        for _ in range(num_hidden_layers):
            num_layer_parameters = 4 * hidden_size * hidden_size + 2 * hidden_size * ffn_dim
            parameters_bytes += num_layer_parameters * num_bytes_per_parameter_layers
        caches_num_element = num_hidden_layers * 2 * sequence_length * batch_size * hidden_size
        caches_bytes = caches_num_element * num_bytes_per_parameter
        largest_bucket_dma_size = parameters_bytes + caches_bytes
        latency_list = latency_collector_all.latency_list
        last_half_latency_list = latency_list[-sequence_length // 2:]
        hbm_bandwidths = [largest_bucket_dma_size / lat for lat in last_half_latency_list]
        avg_hbm_bandwidth_gb_per_sec = sum(hbm_bandwidths) / len(hbm_bandwidths) / 1e9
        *_, test_name = self.id().split('.')
        p0_latency_ms = latency_collector_all.percentile(0) * 1000
        p50_latency_ms = latency_collector_all.percentile(50) * 1000
        p90_latency_ms = latency_collector_all.percentile(90) * 1000
        p95_latency_ms = latency_collector_all.percentile(95) * 1000
        p99_latency_ms = latency_collector_all.percentile(99) * 1000
        p100_latency_ms = latency_collector_all.percentile(100) * 1000

        elapsed = sum(elapsed_list) / len(elapsed_list)
        max_throughput = batch_size * num_new_tokens / min(elapsed_list)
        average_throughput = batch_size * num_new_tokens / elapsed

        hbm_bw_util_percent = avg_hbm_bandwidth_gb_per_sec / (260 * tp_degree) * 100
        report_dict = dict()
        report_dict["Latency P0"] = f'{p0_latency_ms:.1f}'
        report_dict["Latency P50"] = f'{p50_latency_ms:.1f}'
        report_dict["Latency P90"] = f'{p90_latency_ms:.1f}'
        report_dict["Latency P95"] = f'{p95_latency_ms:.1f}'
        report_dict["Latency P99"] = f'{p99_latency_ms:.1f}'
        report_dict["Latency P100"] = f'{p100_latency_ms:.1f}'
        report_dict["Average throughput"] = f'{average_throughput:.1f}'
        report_dict["Peak throughput"] = f'{max_throughput:.1f}'
        report_dict["HBM bandwidth"] = f'{hbm_bw_util_percent:.1f}'

        var_to_name = {
            "p0_latency_ms": "Latency P0",
            "p50_latency_ms": "Latency P50",
            "p90_latency_ms": "Latency P90",
            "p95_latency_ms": "Latency P95",
            "p99_latency_ms": "Latency P99",
            "p100_latency_ms": "Latency P100",
            "average_throughput": "Average throughput",
            "max_throughput": "Peak throughput",
            "hbm_bw_util_percent": "HBM bandwidth",
        }

        report = f'RESULT {test_name}:'
        for key, value in report_dict.items():
            report += f' {key}={value}'
        print(report)

        dtype = dtypes.to_torch_dtype(dtype)
        return generated_sequence, report_dict


class LatencyCollector:

    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)

    def percentile(self, percent):
        latency_list = self.latency_list
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        latency_list = sorted(latency_list)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]