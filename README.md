# Introduction

This repository contains deep learning development environment for contrails project.

# EVA02 installation

Build & install `apex` via following command:

```cd lib && git clone https://github.com/NVIDIA/apex && cd apex && git checkout 2d8302a6c12e202f7b40b13a43daa95f326fd0ea && CC='/usr/bin/gcc-9' CXX='/usr/bin/g++-9' pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./```


# Lib, archetecture and backbones

### HF + Upernet
+ 'openmmlab/upernet-convnext-base'
+ 'google/efficientnet-b5' with no aux head
+ 'facebook/convnextv2-base-22k-224' with no aux head
+ 'facebook/convnext-tiny-224'

+- 'timm/efficientnet_b5.sw_in12k_ft_in1k' safetensor error (does not have a metadata)
+- 'timm/convnextv2_base.fcmae_ft_in22k_in1k_384' safetensor error (does not have a metadata)

- 'timm/maxvit_base_tf_512.in21k_ft_in1k' not implemented