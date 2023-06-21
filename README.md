# Introduction

This repository contains deep learning development environment for contrails project.

# EVA02 installation

Build & install `apex` via following command:

```cd lib && git clone https://github.com/NVIDIA/apex && cd apex && git checkout 2d8302a6c12e202f7b40b13a43daa95f326fd0ea && CC='/usr/bin/gcc-9' CXX='/usr/bin/g++-9' pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./```
