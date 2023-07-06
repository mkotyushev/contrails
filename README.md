# Introduction

This repository contains deep learning development environment for contrails project.

# EVA02 installation

Build & install `apex` via following command:

```cd lib && git clone https://github.com/NVIDIA/apex && cd apex && git checkout 2d8302a6c12e202f7b40b13a43daa95f326fd0ea && CC='/usr/bin/gcc-9' CXX='/usr/bin/g++-9' pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./```


# Lib, archetecture and backbones

### HF + Upernet

Native, full model pretrained:
+ 'openmmlab/upernet-convnext-base'

Native, special cases
+ 'facebook/convnextv2-base-22k-224' with no aux head
+ 'facebook/convnextv2-base-22k-384' with no aux head

Timm backbone:
+ 'maxvit_rmlp_base_rw_384'

Timm backbone (requires patching `transformers`, see below):
+ 'tf_efficientnet_b5'
+ 'tf_efficientnetv2_l'

# Library modifications

TODO: need to be moved to `lib` folder and installed on docker build.

1. `transformers`: 
- /root/miniconda3/envs/contrails/lib/python3.10/site-packages/transformers/models/upernet/modeling_upernet.py: `if self.auxiliary_head is not None` when initializing weights
- /root/miniconda3/envs/contrails/lib/python3.10/site-packages/transformers/models/timm_backbone/modeling_timm_backbone.py: `if hasattr(self._backbone, "return_layers")` before acessing `self._backbone.return_layers`

# Known issues
- efficientnet models tend to significantely slow down for large batch size (e. g. 64 for b5)
- deteministic training is not available for most of the models due to lack of backward kernels for some operations