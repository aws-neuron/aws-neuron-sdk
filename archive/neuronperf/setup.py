import collections
import os
import subprocess

from setuptools import find_packages, setup

# Read __version__.py
version_py = os.path.join("src", "neuronperf", "__version__.py")
with open(version_py, "rt") as fp:
    lines = fp.readlines()
meta = collections.OrderedDict()
for line in lines:
    key, value = line.split("=")
    meta[key.strip()] = value.strip()[1:-1]

# Extract fields for packaging
TITLE = meta["__title__"]
AUTHOR = meta["__author__"]
DESCRIPTION = meta["__description__"]
VERSION = os.getenv("BRAZIL_PACKAGE_VERSION", "0.0.0.0")
LICENSE = meta["__license__"]

# Compute release version and write back meta info for consistency.
GIT_SHA = os.environ.get("BRAZIL_PACKAGE_CHANGE_ID")
if GIT_SHA:
    GIT_SHA = GIT_SHA.strip()[:9]
else:
    # This is probably a local build. Try to attach something meaningful.
    try:
        GIT_SHA = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except:
        GIT_SHA = "0" * 9
VERSION = "{}+{}".format(VERSION.strip(), GIT_SHA)
meta["__version__"] = VERSION
with open(version_py, "wt") as fp:
    for k, v in meta.items():
        fp.write('{} = "{}"\n'.format(k, v))


setup(
    name=TITLE,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="aws neuron",
    packages=find_packages(where="src", exclude=("test",)),
    install_requires=["dill==0.3.4", "numpy", "psutil==5.9.0"],
    python_requires=">=3.6",
    package_dir={"": "src"},
    data_files=[],
    package_data={"": ["py.typed"]},
)
