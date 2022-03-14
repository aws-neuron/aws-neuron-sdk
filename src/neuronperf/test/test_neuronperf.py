# -*- coding: utf-8 -*-

import json
import os
import pathlib
import shutil
import time

import numpy as np
import pytest

import neuronperf


@pytest.mark.sanity
def test_timer():
    timer = neuronperf.Timer()
    with timer:
        time.sleep(1)

    # sanity check
    assert timer.total_duration("s") > 0.5 and timer.total_duration("s") < 1.5

    # check conversions are functional
    assert (
        timer.total_duration("ns")
        > timer.total_duration("us")
        > timer.total_duration("ms")
        > timer.total_duration("s")
    )

    # check timestamp deltas are close to total
    assert timer.total_duration("s") == pytest.approx(timer.durations("s").sum())

    # check iteration functions
    for _ in range(10):
        with timer:
            time.sleep(0.01)
    assert len(timer) > 10

    # check that timer always returns pairs
    timestamps = timer.timestamps()
    for pair in timestamps:
        assert 2 == len(pair)
        assert pair[1] > pair[0]

    # check that len is functional
    assert len(timer) == len(timestamps)


@pytest.mark.sanity
def test_timestamp_convert():
    # test scalar behavior
    assert 1000 == pytest.approx(neuronperf.timestamp_convert(1, "s", "ms"))
    assert 1.5 == pytest.approx(neuronperf.timestamp_convert(1500, "ms", "s"))
    assert 2.3e6 == pytest.approx(neuronperf.timestamp_convert(2.3, "s", "us"))

    # test array behavior
    times = np.array([1, 2, 3])
    times_ms = neuronperf.timestamp_convert(times, "s", "ms")
    assert 1000 == pytest.approx(times_ms[0])


@pytest.mark.sanity
def test_model_index_create_from_file():
    filename = "dummy_model.ext"
    model_name = "dummy"
    index = neuronperf.model_index.create(filename, model_name=model_name)
    assert index["model_name"] == model_name
    assert len(index["model_configs"]) == 1
    assert index["model_configs"][0]["filename"] == filename


@pytest.mark.sanity
def test_model_index_create_delete_save_load():
    filename = "dummy_index.json"
    if os.path.exists(filename):
        neuronperf.model_index.delete(filename)

    model_name = "Dummy"
    model_filename = os.path.join("models", "dummy.model")
    model_index = neuronperf.model_index.create(model_filename, model_name=model_name)
    neuronperf.model_index.save(model_index, filename=filename)
    assert os.path.exists(filename)

    model_index_loaded = neuronperf.model_index.load(filename)
    assert model_index_loaded == model_index
    assert model_index_loaded["model_name"] == model_name
    assert model_index_loaded["model_configs"][0]["batch_size"] == 1

    neuronperf.model_index.delete(filename)
    assert not os.path.exists(filename)


@pytest.mark.sanity
def test_model_index_copy():
    filename = "dummy_index.json"
    if os.path.exists(filename):
        neuronperf.model_index.delete(filename)

    model_filename = os.path.join("models", "dummy.model")
    os.makedirs("models", exist_ok=True)
    pathlib.Path(model_filename).touch()
    model_name = "Dummy"
    model_index = neuronperf.model_index.create(model_filename, model_name=model_name)
    neuronperf.model_index.save(model_index, filename=filename)

    # Test copy API using a pre-loaded model inndex
    neuronperf.model_index.copy(model_index, "new_index.json", "new_models")
    assert os.path.exists("models")
    assert os.path.exists(model_filename)
    assert os.path.exists("new_index.json")
    assert os.path.exists(os.path.join("new_models", "dummy.model"))

    new_index = neuronperf.model_index.load("new_index.json")
    assert new_index["model_configs"][0]["filename"] == os.path.join("new_models", "dummy.model")

    neuronperf.model_index.delete(filename)
    neuronperf.model_index.delete("new_index.json")
    shutil.rmtree("new_models")
    shutil.rmtree("models")


@pytest.mark.sanity
def test_model_index_copy_2():
    filename = "dummy_index.json"
    if os.path.exists(filename):
        neuronperf.model_index.delete(filename)

    model_filename = os.path.join("models", "dummy.model")
    os.makedirs("models", exist_ok=True)
    pathlib.Path(model_filename).touch()
    model_name = "Dummy"
    model_index = neuronperf.model_index.create(model_filename, model_name=model_name)
    neuronperf.model_index.save(model_index, filename=filename)

    # Test copy API using a file
    neuronperf.model_index.copy(filename, "new_index.json", "new_models")
    assert os.path.exists("models")
    assert os.path.exists(model_filename)
    assert os.path.exists("new_index.json")
    assert os.path.exists(os.path.join("new_models", "dummy.model"))

    new_index = neuronperf.model_index.load("new_index.json")
    assert new_index["model_configs"][0]["filename"] == os.path.join("new_models", "dummy.model")

    neuronperf.model_index.delete(filename)
    neuronperf.model_index.delete("new_index.json")
    shutil.rmtree("new_models")
    shutil.rmtree("models")


@pytest.mark.sanity
def test_model_index_move():
    filename = "dummy_index.json"
    if os.path.exists(filename):
        neuronperf.model_index.delete(filename)

    model_filename = os.path.join("models", "dummy.model")
    os.makedirs("models", exist_ok=True)
    pathlib.Path(model_filename).touch()
    model_name = "Dummy"
    model_index = neuronperf.model_index.create(model_filename, model_name=model_name)
    neuronperf.model_index.save(model_index, filename=filename)

    neuronperf.model_index.move(filename, "new_index.json", "new_models")
    assert not os.path.exists(filename)
    assert not os.path.exists(model_filename)
    assert os.path.exists("new_index.json")
    assert os.path.exists(os.path.join("new_models", "dummy.model"))

    new_index = neuronperf.model_index.load("new_index.json")
    assert new_index["model_configs"][0]["filename"] == os.path.join("new_models", "dummy.model")

    neuronperf.model_index.delete("new_index.json")
    shutil.rmtree("new_models")
    shutil.rmtree("models")


@pytest.mark.sanity
def test_model_index_append():
    model_indexes = [
        neuronperf.model_index.create(f"Dummy_{x}", model_name="Dummy") for x in range(10)
    ]
    combined_index = neuronperf.model_index.append(*model_indexes)
    # Assert that combination apparently did happen.
    assert len(combined_index["model_configs"]) == len(model_indexes)
    # Check that batch_sizes haven't been modified.
    assert all(1 == config["batch_size"] for config in combined_index["model_configs"])

    # Test for duplicate filtering behavior
    model_indexes = [neuronperf.model_index.create("Dummy") for _ in range(10)]
    combined_index = neuronperf.model_index.append(*model_indexes)
    assert len(combined_index["model_configs"]) == 1


@pytest.mark.sanity
def test_model_index_filter():
    idx_1 = neuronperf.model_index.create("fake", performance_level=2, compile_s=1)
    idx_2 = neuronperf.model_index.create("fake2", compile_s=2)
    idx = neuronperf.model_index.append(idx_1, idx_2)

    filtered = neuronperf.model_index.filter(idx, filename="fake")
    print(filtered)
    assert 1 == len(filtered["model_configs"])
    assert "fake" == filtered["model_name"]

    filtered = neuronperf.model_index.filter(idx, performance_level=2)
    assert 1 == len(filtered["model_configs"])
    assert "fake" == filtered["model_name"]

    # None key should filter nothing
    filtered = neuronperf.model_index.filter(idx, compile_s=None)
    assert 2 == len(filtered["model_configs"])


@pytest.mark.sanity
@pytest.mark.slow
def test_benchmarker():
    dummy_model = lambda x: None
    dummy_load = lambda path, device_id: dummy_model
    b = neuronperf.benchmarking.Benchmarker(
        id=0, device_id=0, load_fn=dummy_load, model_filename="test", inputs=[], workers_per_model=2
    )
    b.start()
    time.sleep(1.5)
    b.stop()

    assert b.status == "finished"
    assert all(n_infs > 100 for n_infs in b.n_infs)


@pytest.mark.slow
def test_benchmark_multithread():
    benchmarker_results = neuronperf.cpu.benchmark(
        neuronperf.DummyModel,
        [np.array([1, 2, 3, 4])],
        duration=2,
        n_models=4,
        multiprocess=False,
        multiinterpreter=False,
        verbosity=2,
        return_timers=True,
    )

    # Return value is a list of tuples:
    # [(config, results), (config, results), ...]
    # Each config is a dict. Each result is a dict.

    # A single configuration without workers_per_model set will produce 2 results
    assert len(benchmarker_results) == 2

    for benchmarker_result in benchmarker_results:
        config, results = benchmarker_result
        assert "cpu_percents" in results
        assert "mem_percents" in results
        assert not config["multiprocess"]
        assert not config["multiinterpreter"]
        assert results["status"] == "finished"
        assert results["n_infs"] > 100


@pytest.mark.slow
def test_benchmark_multithread_2():
    dummy_model = lambda x: None
    dummy_load = lambda path, device_id: dummy_model
    reports = neuronperf.benchmark(
        load_fn=dummy_load,
        model_filename="dummy_filename",
        inputs=[[1]],
        duration=2,
        n_models=4,
        multiprocess=False,
        multiinterpreter=False,
        verbosity=2,
    )

    # A single configuration without workers_per_model set will produce 2 results
    assert len(reports) == 2
    report = reports[0]
    assert not report["multiprocess"]
    assert not report["multiinterpreter"]
    assert report["status"] == "finished"
    assert report["total_infs"] > 100


@pytest.mark.slow
def test_benchmark_multiprocess():
    n_models = 16
    benchmarker_results = neuronperf.cpu.benchmark(
        neuronperf.DummyModel,
        inputs=[np.array([1, 2])],
        batch_sizes=[1],
        duration=2,
        n_models=n_models,
        multiprocess=True,
        multiinterpreter=False,
        verbosity=2,
        return_timers=True,
    )

    # A single configuration will produce a single result tuple
    assert len(benchmarker_results) == 2
    # Extract the benchmarker results
    config, results = benchmarker_results[0]
    # Confirm that there is least 1 timer / model for each benchmarker
    assert len(next(iter(results["timers"].values()))) >= n_models
    assert config["multiprocess"]
    assert not config["multiinterpreter"]
    assert results["status"] == "finished"
    assert results["n_infs"] > 100


@pytest.mark.slow
def test_benchmark_multiinterpreter():
    benchmarker_results = neuronperf.cpu.benchmark(
        neuronperf.DummyModel,
        inputs=[np.array([1, 2])],
        duration=2.5,
        n_models=2,
        multiprocess=False,
        multiinterpreter=True,
        verbosity=2,
        return_timers=True,
    )

    # A single configuration without workers_per_model set will produce 2 results
    assert len(benchmarker_results) == 2
    # Extract the benchmarker results
    config, results = benchmarker_results[0]
    assert config["multiinterpreter"]
    assert results["status"] == "finished"
    assert results["n_infs"] > 100


@pytest.mark.slow
def test_reporting():
    benchmarker_results = neuronperf.cpu.benchmark(
        neuronperf.DummyModel,
        inputs=[np.array([1, 2, 3, 4])],
        n_models=[1, 4],
        duration=2,
        verbosity=2,
        return_timers=True,
    )

    assert len(benchmarker_results) == 4
    reports = neuronperf.get_reports(benchmarker_results)
    assert len(reports) == len(benchmarker_results)
    assert all("total_infs" in report for report in reports)

    neuronperf.print_reports(reports)
    csv_file = neuronperf.write_csv(reports)
    os.remove(csv_file)
    json_file = neuronperf.write_json(reports)
    with open(json_file, "rt") as fp:
        json.load(fp)
    os.remove(json_file)
