# -*- coding: utf-8 -*-

"""
neuronperf.model_index
~~~~~~~~~~~~~~~~~~~~~~~
Provides utilities for working with model indexes.
"""

from typing import Any, List, Union

import builtins
import copy as copy_module
import itertools
import json
import logging
import os
import pathlib
import random
import shutil


from .__version__ import __version__
from .compile_constants import FAST_MATH_OPTIONS


log = logging.getLogger(__name__)

MODEL_INDEX_SUFFIX = ".json"


def generate_id(length: int = 8):
    """Generate a random-enough sequence to append to model names and prevent collisions."""
    id_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    new_id = [id_chars[random.randrange(len(id_chars))] for _ in range(length)]
    return "".join(new_id)


def generate_name(model_name: str):
    """Generate a model index name from a model name."""
    return model_name + "_" + generate_id() + MODEL_INDEX_SUFFIX


def _create(model_name: str, compile_info: list) -> dict:
    if not isinstance(compile_info, list):
        log.exception(
            "Expected a list of compile info dicts, received '{}'.".format(str(type(compile_info)))
        )
    model_index = {
        "NeuronPerf_version": __version__,
        "model_name": model_name,
        "model_configs": compile_info,
    }
    return model_index


def create(
    filename: str,
    model_name: str = None,
    batch_size: int = 1,
    pipeline_size: int = 1,
    performance_level: int = max(FAST_MATH_OPTIONS),
    compile_s: float = None,
    status: str = "finished",
) -> dict:
    r"""
    Create a new model index from a pre-compiled model.

    :param str filename: The path to the compiled model.
    :param str model_name: A friendly name for the model. Will default to filename.
    :param int batch_size: The batch size at compilation for this model.
    :param int pipeline_size: The pipeline size used at compilation for this model.
    :param int performance_level: The performance level this model was compiled with.
    :param float compile_s: Seconds spent compiling.
    :param str status: A string describing compilation result. Can be "finished" or "error".
    :return: A new dictionary representing a model index.
    :rtype: dict
    """
    if not model_name:
        model_name = filename
    compile_info = [
        {
            "filename": filename,
            "batch_size": batch_size,
            "pipeline_size": pipeline_size,
            "performance_level": performance_level,
            "compile_s": compile_s,
            "status": status,
        }
    ]
    return _create(model_name, compile_info)


def delete(filename: str):
    """Deletes the model index and all associated models referenced by the index."""
    if not os.path.exists(filename):
        log.warning("Asked to delete '{}', but it can't be located.".format(filename))
        return

    # Load the index
    configs = load(filename)["model_configs"]

    # Remove all referenced models
    model_filenames = map(lambda x: x["filename"], itertools.chain(configs))
    for model_filename in model_filenames:
        log.debug(f"Deleting '{model_filename}'.")
        if os.path.exists(model_filename):
            if os.path.isdir(model_filename):
                shutil.rmtree(model_filename)
            else:
                os.remove(model_filename)

    # Finally, remove the model index itself
    log.debug(f"Deleting '{filename}'")
    os.remove(filename)


def copy(old_index: Union[str, dict], new_index: str, new_dir: str) -> str:
    r"""
    Copy an index to a new location. Will rename ``old_index``
    to ``new_index`` and copy all model files into ``new_dir``,
    updating the index paths.

    This is useful for pulling individual models out of a pool.

    Returns the path to the new index.
    """
    os.makedirs(new_dir, exist_ok=True)
    index = _sanitize(old_index)[0].copy()

    configs = index["model_configs"]
    for config in configs:
        path = pathlib.Path(config["filename"])
        config["filename"] = str(shutil.copy2(path, new_dir))

    return save(index, new_index)


def move(old_index: str, new_index: str, new_dir: str) -> str:
    """This is the same as ``copy`` followed by ``delete`` on the old index."""
    index = copy(old_index, new_index, new_dir)
    delete(old_index)
    return index


def _sanitize(*model_indexes: Union[str, dict]) -> List[dict]:
    r"""
    Helper function to load indexes if strings are provided.
    If already loaded, this is a no-op.
    """
    if not model_indexes:
        raise ValueError("No model indexes were provided.")
    indexes = []
    # Load any paths provided and sanity check all inputs.
    for index in model_indexes:
        if not index:
            raise ValueError("An empty value was received, but expected a model index.")
        if isinstance(index, str):
            index = load(index)
        if not isinstance(index, dict):
            raise TypeError("Expected a model index, but received '{}'.".format(str(type(None))))
        if not len(index) > 0:
            raise ValueError("Received an empty model index.")
        indexes.append(index)
    # Check versions are all the same, and emit a warning if they aren't.
    versions = set(map(lambda x: x["NeuronPerf_version"], indexes))
    if len(versions) > 1:
        log.warning("Received model with different versions: '{}'.".format(str(versions)))
    model_name = indexes[0]["model_name"]
    # Ensure model names are matching.
    if not all(model_name == index["model_name"] for index in indexes):
        model_names = list(set(map(lambda x: x["model_name"], indexes)))
        log.warning("Received model indexes with different model names: {}".format(model_names))
    return indexes


def append(*model_indexes: Union[str, dict]) -> dict:
    r"""
    Appends the model indexes non-destructively into a new model index, without
    modifying any of the internal data.

    This is useful if you have benchmarked multiple related models and wish to
    combine their respective model indexes into a single index.

    Model name will be taken from the first index provided.
    Duplicate configs will be filtered.

    :param Union[str, dict] model_indexes: Model indexes or paths to model indexes to combine.
    :return: A new dictionary representing the combined model index.
    :rtype: dict
    """
    indexes = _sanitize(*model_indexes)
    # Extract the model configs from the indexes
    config_iter = map(lambda index: copy_module.deepcopy(index["model_configs"]), indexes)
    # Combine the model configs
    combined = list(itertools.chain.from_iterable(config_iter))
    # Split unique and duplicate configs
    duplicate = []
    unique = []
    for config in combined:
        if config in unique:
            duplicate.append(config)
        else:
            unique.append(config)
    if len(duplicate) > 0:
        log.warning(
            (
                f"There were {len(duplicate)} duplicate model configs "
                "filtered. The duplicates were:\n"
                "{}".format("\n".join(map(lambda c: str(c), duplicate)))
            )
        )
    # Build new index from configs
    return _create(indexes[0]["model_name"], unique)


def save(model_index: dict, filename: str = None, root_dir=None) -> str:
    r"""Save a NeuronPerf model index to a file."""
    if not filename:
        model_name = model_index["model_name"]
        filename = generate_name(model_name)
    if not filename.lower().endswith(MODEL_INDEX_SUFFIX):
        filename += MODEL_INDEX_SUFFIX
    if not root_dir:
        root_dir = "."
    try:
        with open(os.path.join(root_dir, filename), "w") as fp:
            json.dump(model_index, fp)
    except OSError:
        log.exception("Failed to write '{}'.".format(filename))
    return filename


def load(filename) -> dict:
    """Load a NeuronPerf model index from a file."""
    model_index = None
    try:
        with open(filename, "r") as fp:
            model_index = json.load(fp)
    except OSError:
        # file is probably not a model index
        log.exception("Failed to load model index '{}'".format(filename))
    else:
        from distutils.version import LooseVersion

        try:
            if LooseVersion(model_index["NeuronPerf_version"]) > LooseVersion(__version__):
                log.warning(
                    "Model index newer than NeuronPerf (version {} > {}). Try updating NeuronPerf.".format(
                        model_index["NeuronPerf_version"], __version__
                    )
                )
        except TypeError:
            log.warning(
                "Couldn't compare model index version ({}) to NeuronPerf version ({}), continuing anyway.".format(
                    model_index["NeuronPerf_version"], __version__
                )
            )

    return model_index


def filter_configs(configs, filter_name, filter_values) -> List:
    """Filters provided configs on specified filter and value and returns a new config list."""
    if filter_values is None:
        return configs.copy()
    # Filter on configs that have the filter_name and value is in filter_values
    if not isinstance(filter_values, list):
        filter_values = [filter_values]
    return list(
        builtins.filter(
            lambda config: filter_name in config and config[filter_name] in filter_values, configs
        )
    )


def filter(index: Union[str, dict], **kwargs) -> dict:
    r"""
    Filters provided model index on provided criteria and returns a new index.
    Each kwarg is a standard (k, v) pair, where k is treated as a filter name
    and v may be one or more values used to filter model configs.
    """
    index = _sanitize(index)[0].copy()

    # Filter each config on provided kwargs pairs.
    configs = index["model_configs"]
    for k, v in kwargs.items():
        configs = filter_configs(configs, k, v)

    index["model_configs"] = configs
    return index
