# OASIS-2 Pair Generation

## Purpose

This folder contains the OASIS-2-specific pair generator used as an example
dataset adapter for the registration benchmark.

The benchmark itself is dataset-agnostic. This adapter only creates a CSV with:

```text
fixed_path,moving_path
```

An optional `pair_id` column may also be included.

---

## Dataset Layout

The raw OASIS-2 data is organized as session folders:

```text
OAS2_RAW_PART1/
  OAS2_0001_MR1/
    RAW/
      mpr-1.nifti.hdr
      mpr-1.nifti.img
      mpr-2.nifti.hdr
      mpr-2.nifti.img
      ...
```

For each valid session, the adapter creates one monomodal intra-session pair:

```text
fixed  = RAW/mpr-1.nifti.hdr
moving = RAW/mpr-2.nifti.hdr
```

The CSV stores the `.hdr` path and requires the matching `.img` file to exist.

---

## Usage

From `benchmarks/registration`:

```powershell
python datasets/oasis2/make_pairs.py `
  --root /path/to/OAS2_RAW_PART1/OAS2_RAW_PART1 `
  --out data/oasis2_pairs.csv
```

OASIS-2 data access and download links are available from the
[official OASIS-2 page](https://sites.wustl.edu/oasisbrains/home/oasis-2/).

---

## Citations

Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner,
R. L. (2010). Open Access Series of Imaging Studies (OASIS): Longitudinal MRI
data in nondemented and demented older adults. *Journal of Cognitive
Neuroscience, 22*, 2677-2684. https://doi.org/10.1162/jocn.2009.21407
