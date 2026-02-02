# FORCE: Fast Orientation Reconstruction and Compartment Estimation

## Overview

FORCE is a dictionary-based diffusion MRI reconstruction method that uses
pre-simulated signal dictionaries to estimate microstructural parameters
from diffusion-weighted imaging data.

## Module Structure

- `force.py` - Main simulation and signal generation API
- `_force_core.pyx` - Cython implementation of core signal generation
- `_multi_tensor_omp.pyx` - OpenMP-accelerated multi-tensor signal computation

## Features

- Multi-compartment tissue simulation (WM, GM, CSF)
- Configurable diffusivity ranges
- Bingham distribution-based fiber orientation dispersion
- Support for 1-3 fiber populations per voxel
- microFA computation
- DTI and DKI metric generation

## Usage

```python
from dipy.sims.force import generate_force_dictionary

# Generate simulation dictionary
signals, labels, params = generate_force_dictionary(
    gtab,
    num_simulations=100000,
    num_cpus=8
)
```

## References

- FORCE methodology paper (in preparation)
