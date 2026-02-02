# FORCE Reconstruction Module

## Overview

This module provides the reconstruction/matching component of FORCE
(Fast Orientation Reconstruction and Compartment Estimation).

It performs dictionary-based matching between real diffusion MRI signals
and pre-simulated signal dictionaries to estimate:

- Fiber orientations (up to 3 fibers per voxel)
- Tissue fractions (WM, GM, CSF)
- Microstructural parameters (FA, MD, RD, microFA)
- Orientation dispersion
- DKI metrics (AK, RK, MK, KFA)

## Module Structure

- `force.py` - Main reconstruction API
- `_force_search.pyx` - High-performance vector search implementation

## Features

- Fast inner-product based similarity search
- Chunk-based parallel processing
- Memory-efficient memmap support
- Configurable penalty for fiber complexity
- Integration with DIPY's PAM format

## Usage

```python
from dipy.reconst.force import FORCEModel

# Create model with pre-computed dictionary
model = FORCEModel(gtab, dictionary_path='simulated_data.npz')

# Fit to data
fit = model.fit(data, mask=brain_mask)

# Access results
fa_map = fit.fa
peaks = fit.peak_dirs
```
