# Processed datasets

Each dataset contains exactly three files:

- `data.dat`: headerless, C-order float32 data with shape `(time, node, feature)`.
- `desc.json`: feature names, shape, sampling resolution, and recommended experiment settings.
- `adj_mx.pkl`: a numeric adjacency matrix in the same node order as `data.dat`.

Binary files are tracked with Git LFS. `manifest.json` records the byte size and SHA-256 checksum of all 30 files. Run `python scripts/verify_datasets.py` from the repository root to validate the local data.

## Inventory

| Dataset directory | Minutes per step | Shape (T x N x F) | Size (MiB) |
| --- | ---: | --- | ---: |
| MetroFlow | 10 | 7044 x 302 x 14 | 114.31 |
| NJRC_10min | 10 | 58032 x 36 x 28 | 223.16 |
| NJRC_15min | 15 | 38688 x 36 x 28 | 148.78 |
| NJRC_5min | 5 | 116064 x 36 x 28 | 446.30 |
| NYC | 30 | 4368 x 98 x 9 | 14.73 |
| PEMS03 | 5 | 26208 x 358 x 4 | 143.65 |
| SZparking_10min | 10 | 4176 x 189 x 25 | 75.54 |
| SZparking_15min | 15 | 2784 x 189 x 25 | 50.45 |
| SZparking_5min | 5 | 8352 x 189 x 25 | 150.81 |

Total size is approximately 2.06 GiB. NJRC includes the additional 3-minute resolution supplied with the project. The default experiment configurations use NJRC at 10 minutes, SZparking at 10 minutes, MetroFlow at 10 minutes, NYC at 30 minutes, and PEMS03 at 5 minutes.

## Loading

```python
import json
import numpy as np

name = "NJRC_10min"
with open(f"datasets/{name}/desc.json", encoding="utf-8") as stream:
    description = json.load(stream)
data = np.memmap(f"datasets/{name}/data.dat", dtype="float32", mode="r",
                 shape=tuple(description["shape"]))
print(data.shape)
```

Use NumPy 2 as specified in `requirements.txt` for the bundled NumPy adjacency pickles. The DSFNet architecture does not require an adjacency matrix in its forward pass; these matrices are retained as part of the processed datasets.

## Features used by the supplied models

Indices below refer to the original feature axis in `data.dat`, using zero-based indexing. Complete feature names appear in each dataset's `feature_description` list.

| Configuration | Traffic input and target columns | Exogenous input columns |
| --- | --- | --- |
| NJRC | 0, 2, 3, 4, 5, 6, 9, 10, 11, 12 | 13, 16, 25, 26, 27 |
| SZparking | 0 through 9 | 12, 15, 22, 23, 24 |
| MetroFlow | 0 through 7 | 8 through 13 |
| NYC | 0 through 5 | 6 through 8 |
| PEMS03 | 0 | 1 through 3 |

Weather classes and calendar indices are already prepared in the data. NJRC and SZparking embed two weather-class columns and three calendar columns. MetroFlow embeds three weather-class columns and three calendar columns. NYC and PEMS03 use three calendar columns.

The chronological split ratios are read from `regular_settings` in `desc.json`. The dataset reader assigns `floor(T * validation_ratio)` steps to validation and `floor(T * test_ratio)` steps to testing; the remaining steps are used for training. Each split produces sliding windows with 12 input and 12 output steps. The default datasets are long enough to avoid overlap between splits.

## Data contents and release scope

The release contains processed numeric traffic aggregates and covariates. Raw vehicle or trip records, local spreadsheets, and standalone site-information tables are excluded. Some retained covariates describe site geography or infrastructure, as documented in `desc.json`; the processed arrays are preserved without changing their values or node order.

Dataset source and redistribution terms were not recorded in the supplied descriptions. No new dataset license or provenance claim is assigned by this package.
