# Implementation Details: Local Compute Script 1

**Script name:** `1_local_compute_all.py`

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [Overview](#overview)
  - [High-level Flow (Per AEZ)](#high-level-flow-per-aez)
- [Dependencies and Environment](#dependencies-and-environment)
  - [External Libraries](#external-libraries)
  - [Project-Local Utilities](#project-local-utilities)
- [Data Model and File Layout](#data-model-and-file-layout)
  - [`statuscsv` Schema](#statuscsv-schema)
- [Stage-by-Stage Documentation](#stage-by-stage-documentation)
  - [1. AEZ Sampling & Representative Tile Selection](#1-aez-sampling--representative-tile-selection)
  - [2. High-Resolution HR Imagery Acquisition & Chunking](#2-high-resolutionhr-imagery-acquisition--chunking)
  - [3. FracTAL-ResUNet Inference and Stitching](#3-fractal-resunet-inference-and-stitching)
  - [4. Feature Extraction for “Easy Positives”](#4-feature-extraction-for-easy-positives)
  - [5. Plantation Detection via YOLO](#5-plantation-detection-via-yolo)
  - [6. AEZ-Level Joins and Export](#6-aez-level-joins-and-export)
- [Orchestration & Resilience](#orchestration--resilience)
- [Coordinate Systems and Resolutions](#coordinate-systems-and-resolutions)
- [Parameters](#parameters)
- [Inputs and Outputs (Per AEZ)](#inputs-and-outputs-per-aez)
- [Minimal “How to Run”](#minimal-how-to-run)

</details>

---

## Overview

This code implements the first part of the three-stage methodology:

1. Tiling of AEZ regions and sampling representative tiles  
2. High-resolution (≈1.19 m/pixel at zoom 17) image acquisition  
3. Farm/non-agro instance segmentation using a FracTAL-ResUNet model  
4. Geometric/texture feature computation (entropy, rectangularity, size) to identify “easy positives”  
5. High-precision plantation masks and boundary generation via a fine-tuned YOLO model  
6. Vectorization and export of boundaries  

Orchestration is per-AEZ with resumability via a lock-safe `status.csv`.

---

### High-level Flow (Per AEZ)

- **Pre-process:** Obtain representative tiles approximating AEZ’s embedding distribution (K-means K=32 + JS-divergence selection ≈3%).  
- **Download imagery:** Chunk into 16×16 patches (256×256 px each).  
- **Run FracTAL-ResUNet inference** → stitch logits/boundary maps.  
- **Instance segmentation (InstSegm)** → labeled instances.  
- **Feature extraction:** Entropy, rectangularity, size.  
- **YOLO plantation detection** → stitched binary mask.  
- **Vectorization & export:** CRS normalization + AEZ-level zipping.

---

## Dependencies and Environment

### External Libraries

`mxnet (Gluon)`, `ultralytics (YOLO)`, `PIL/Pillow`, `opencv-python (cv2)`, `numpy (np)`, `scikit-image (measure/filters/morphology)`, `geopandas`, `GDAL/OGR/OSR`, `pandas`, `scipy (Jensen–Shannon)`, `filelock`, `pydrive2`, `earthengine-api (ee)`, `multiprocessing`.

```python
# Allow large images in Pillow
PIL.Image.MAX_IMAGE_PIXELS = 2000000000
```
We set Pillow’s pixel cap to allow opencv to process larger segments while calculating entropy

### Project-Local Utilities

**Required files:**
- `commons.py`
- `dataset.py`
- `instance_segment.py`

`commons.py` is copied from the **samgeo** library (to control parallel downloads = 1 to avoid network error when parallel download is allowed).  
`decode` and `instance_segment.py` come from https://github.com/waldnerf/decode.

**Utilities:**
- `commons.tms_to_geotiff(...)`: fetches TMS satellite tiles → GeoTIFF.  
- `commons.raster_to_shp(...)`: raster → vector polygonization.  
- `decode/FracTAL_ResUNet/...`: FracTAL-ResUNet code + weights.  
- `instance_segment.InstSegm`: post-inference segmentation helper.

---

## Data Model and File Layout

Each AEZ has a directory: `AEZ_<id>/`. We used status.csv to maintain state for each AEZ on how much data is downloaded and processed and the boundaries are ready. The schema at tile level and AEZ level is given below.


### Structure

| File / Folder | Description |
|----------------|-------------|
| `status.csv` | per-tile processing ledger |
| `field.tif` | HR mosaic for tile bbox |
| `chunks/` | 256×256 patch tiles (`chunk_i_j.tif`) |
| `logits_bounds.pickle` | model outputs per chunk |
| `instance_predicted.pickle` | stitched instance labels |
| `all.shp` | merged polygons + attributes |
| `plantation.shp` | polygons detected by YOLO |
| `<AEZ>_boundaries_*.shp` | joined shapefiles (≤2GB each) |
| Zipped archives | ready for delivery |

Each AEZ tile is processed individually, then merged into chunked AEZ shapefiles (≤2 GB limit).

---

### `statuscsv` Schema

| Column | Description |
|---------|--------------|
| `index` | Tile ID (0..N-1) |
| `points` | ((lat1, lon1), (lat2, lon2)) top-left / bottom-right |
| `overall_status` | Tile finished end-to-end |
| `download_status`| Boolean flags per stage |
|`model_status` |Boolean flags per stage |
|`segmentation_status`|Boolean flags per stage |
|`postprocessing_status`|Boolean flags per stage |
|`plantation_status`| Boolean flags per stage |

Atomic file updates are handled by `FileLock` to prevent concurrent access conflicts.
#### Description on why we use Filelock
We use file locks as when we parallely process different tiles at different state, the status.csv act as a common ledger. Now it can happen than two different process read and write onto csv at the same time. Hence it will write wrong information as both the processes wont be able to write other’s processed work, and can go into a cycle.

---

## Stage-by-Stage Documentation

### 1. AEZ Sampling & Representative Tile Selection

**Functions:**  
`get_tiles(roi)`, `get_histogram(roi, tiles_path)`, `get_representative_tiles(directory)`, `pre_process(roi, directory)`

**Inputs:**  
- AEZ ROI (GEE FeatureCollection filtered by `ae_regcode`)  
- Google Satellite Embedding V1 (64-dim, annual)  
- LULC mask (vegetated range 6–12)

**Procedure:**
| Sr No: | Step | Description| Comments|
|---------|--------------|-------------|-----------|
|1.| **Gridding:**| Generate zoom-17 tiles aligned to TMS.| These tiles are equivalent to 32X32 images of 256X256 pixels at the given zoom. We  writes tiles to GEE (write_to_gee) as it is faster. We take the max bounds of the ROI region, create grids and clip it to the ROI taking only the completed tiles.|  
|2.| **Clustering:**| Sample ≤50k pixels → WekaKMeans (k=32).| |  
|3.| **Per-tile histogram:**| `reduceRegions` with frequency histogram.|It computes frequency histogram of 32 different classes found by KMeans algorithm. The frequency histogram is computed for each tile and saved as CSV in the next step so that we can process them locally. |  
|4.| **Export:**| Drive → local via `pydrive2`.| to Drive as CSV; pull locally via pydrive2. |
|5.| **Greedy selection:**| Minimize JS divergence; select ≈3% of tiles.| compute full AEZ distribution; greedily select p tiles (default p=60, ≈3%) minimizing Jensen–Shannon divergence between the subset’s aggregated distribution and full AEZ distribution. We can read more about this in paper.|
|6.| **Init `status.csv`:**| Split 32×32 grid → four 16×16 grids.|initialize rows for the selected tiles. Here 32X32 grid is further broken down to four 16X16 grids. This is done as the rest of the pipeline is built for 16X16 size grid as doing more size is not efficient and gives memor out range error. |

**Tunables:**  
`k=32`, `numPixels=50000`, `p≈3%`, `tileScale`,for reduceRegions if ever we get out of memory error, vegetated mask range if we get more classes and wants to do only on those.

---

### 2. High-Resolution (HR) Imagery Acquisition & Chunking

**Functions:**  
`download(...)`, `tms_to_geotiff(...)`, `divide_tiff_into_chunks(...)`

**Inputs:**  
Tile bbox in lat/lon, zoom=17, chunk_size=256

**Procedure:**
1. If `download_status=False`, download via `tms_to_geotiff(...)`.  
2. Save HR mosaic → `field.tif`.  
3. Split into 256×256 patches → `chunks/`.  
4. Mark `download_status=True`.

> Zoom=17 → ≈1.19 m/pixel.  
> Deterministic chunk naming preserves (i,j) indices.

---

### 3. FracTAL-ResUNet Inference and Stitching

**Functions:**  
`load_model()`, `run_inference(...)`, `run_model(...)`

**Model:**  
`FracTAL_ResUNet_cmtsk (nfilters_init=32, depth=6, NClasses=1)`  
Weights: `india_Airbus_SPOT_model.params`

**Procedure:**
1. Load dataset using `Planet_Dataset_No_labels`.  
2. Run inference (`batch_size=32`, GPU id=0).  
3. Save to `logits_bounds.pickle`.  
4. Mark `model_status=True`.

Note:We save every output of every stage so that if we get an error at any point we don’t have to repeat the entire process all over again. So the model output is saved as logits_bounds.pickle and we write model_status=True for that tile.


**Segmentation:**
- Stitch logits + bounds.  
- Thresholds: `t_ext=0.3`, `t_bnd=0.4`.  
- `InstSegm` → labeled instance map.  
- Threshold tuning adjusts boundary sharpness.

Note:These values are set to 0.3 and 0.4 as they give the best results for the boundaries. Varying them sometimes produces better results in specific cases. For example if we increase t_bnd then it could recognise thinner boundaries as well but gives us very thick boundaries for normal boundary pixel. This InstSegm will create segments from the logits. These segments are the actual boundaries. 

---

### 4. Feature Extraction for “Easy Positives”

**Functions:**  
`get_min_max_array(...)`, `process_in_chunks(...)`, `save_field_boundaries(...)`

**Purpose:**  
Compute entropy, rectangularity, and size per instance.

**Optimizations:**
Bounding box caching for O(1) segment lookup.  
Parallel extraction via `multiprocessing.Pool(12)`.

Note: process_in_chunks(...) partitions instance ids and uses multiprocessing.Pool(12) calling map_entropy to produce (id, ent, rect, size) tuples. We use 12 as our production machine has 12 cores


Results are assembled into a pandas DataFrame and merged back to vectors.

**Features:**
- **Entropy:** `disk(5)`, mean over pixels > 5.2.  
- **Rectangularity:** area / min-rectangle area.  
- **Size:** pixel count.

**Vectorization Workflow:**
1. Write raster → GeoTIFF.  
2. Polygonize → `all.shp`.  
3. Reproject EPSG:3857 → EPSG:4326.  
4. Merge attributes (id, ent, rect, size).  
5. Mark `postprocessing_status=True`.
6. Note:
It computes bounding boxes for each instance id; global variables store the HR image and instance map for parallel extraction. This function is written to increase the computation speed. Normally in numpy if we want a subarray with certain values inside a very large array, it takes n^2 time as it will read the entire array row by row one value at a time until it finds those elements. So if there are p different subarrays to be found inside a very big array, it takes pXn^2 time which is a lot of time. To reduce it we traverse the very large array once and store the top left and bottom right (least and highest) indexes of each different segment which is a connected component into a dictionary. Then for searching a segment we use this dictionary which gives us object in order 1. Hence the order to process p segment is p.
---

### 5. Plantation Detection via YOLO

**Functions:**  
`run_plantation_model(...)`, `process_image(...)`, `stitch_masks(...)`

**Model:**  
`plantation_model.pt` (Ultralytics YOLO)  
Class: `'plantations'`  
Confidence: `0.5`

**Procedure:**
1. YOLO segmentation per chunk.  
2. OR masks → full-tile plantation mask.  
3. Vectorize → `plantation.shp`.  
4. Mark `plantation_status=True`.

---

### 6. AEZ-Level Joins and Export

**Functions:**  
`join_boundaries(...)`, `join_boundaries_for_domain(...)`, `zip_vector(...)`

**Procedure:**
1. Concatenate tiles in batches of ~100 (≤2 GB).  
2. Merge into `<AEZ>_boundaries_<batch>.shp`.  
3. Zip for delivery.  
4. Write sentinel file `all_done`.

---

## Orchestration & Resilience

**Entrypoint:**  
`__main__` iterates AEZ_no ∈ {1..20}

1. Resolve ROI from GEE (`users/mtpictd/agro_eco_regions`).  
2. `pre_process(...)` → generate `status.csv`.  
3. `run(...)` executes incomplete tiles:  
   `download → run_model → get_segmentation → run_postprocessing → run_plantation_model`  
4. Run `join_boundaries` post completion.  
5. Retry logic with exponential backoff, resuming via `status.csv`.  
6. Logging (optional): `<AEZ>/output.log`, used by Celery.

---

## Coordinate Systems and Resolutions

| Item | CRS / Resolution |
|------|------------------|
| HR rasters | EPSG:3857 (Web Mercator) |
| Vector deliverables | EPSG:4326 (WGS 84) |
| HR scale | zoom 17 ≈ 1.19 m/pixel |
| Chunk size | 256×256 ≈ 305 m² |
| Embedding scale | 10 m (Google Satellite Embedding V1 Annual) |

---

## Parameters

| Component | Parameter | Default | Purpose |
|------------|------------|----------|----------|
| K-Means | K | 32 | Embedding clusters per AEZ |
| JS selection | p | 60 | Tiles (~3 %) minimizing JS divergence |
| reduceRegions | tileScale | 2 | Memory/performance trade-off |
| HR download | zoom | 17 | ~1.19 m/pixel |
| Chunking | chunk_size | 256 | Patch size for inference |
| FracTAL | batch_size | 32 | Inference throughput |
| InstSegm | t_ext | 0.3 | Extent threshold |
| InstSegm | t_bnd | 0.4 | Boundary threshold |
| Entropy | disk radius | 5 | Texture neighborhood |
| Entropy cutoff | > 5.2 | Pixels contributing to mean |
| YOLO | conf(plantations) | 0.5 | High-precision keep |
| MP | pool size | 12 | Feature extraction parallelism |

---

## Inputs and Outputs (Per AEZ)

### Inputs
- AEZ boundary (`users/mtpictd/agro_eco_regions` with `ae_regcode`)  
- GEE assets: Satellite Embedding V1 (ANNUAL), LULC mask (`pan_india_lulc_v3_2023_2024`)  
- Model weights:  
  - `india_Airbus_SPOT_model.params` (FracTAL)  
  - `plantation_model.pt` (YOLO)

### Outputs
- `all.shp`: instance polygons + attributes (id, ent, rect, size)  
- `plantation.shp`: plantation polygons  
- `<AEZ>_boundaries_*.shp` + zipped archives

---

## Minimal “How to Run”

```bash
# 1) Configure environment and credentials:
ee.Authenticate()
ee.Initialize(project='raman-461708')

# Google Drive credentials:
# client_secrets.json + credentials.json present

# Model weights:
# india_Airbus_SPOT_model.params, plantation_model.pt

# 2) Ensure project-local utils are on PYTHONPATH:
# decode/FracTAL_ResUNet/... , instance_segment.py , commons.py

# 3) Execute:
python main.py

# The script iterates AEZ 1..20:
#   - Pre-process to select representative tiles (status.csv)
#   - Download HR, run FracTAL, segment, compute features, vectorize
#   - Run YOLO plantations, vectorize
#   - Join per-AEZ outputs and zip
```
