# Documentation: Boundary Generation at Block Level

**Script:** `Scrubland-Field-Delineation/src (Block level)/script.py`

---

<details>
<summary><strong>📑 Table of Contents</strong></summary>

- [Overview](#overview)
- [Method: get_points(roi, directory)](#method-get_pointsroi-directory)
  - [Purpose](#purpose)
  - [Signature](#signature)
  - [Parameters](#parameters)
  - [Returns](#returns)
  - [What It Creates on Disk](#what-it-creates-on-disk)
- [Tiling Scheme & Geometry Math](#tiling-scheme--geometry-math)
  - [Coordinate Space](#coordinate-space)
  - [Steps](#steps)
- [CSV Schema](#csv-schema)
- [How the CSV Is Used](#how-the-csv-is-used)
- [Directory Structure (End-to-End)](#directory-structure-end-to-end)
- [Behavior Details & Edge Cases](#behavior-details--edge-cases)

</details>

---

## Overview

This script performs **block-level boundary generation**, identical to `1_local_compute_all.py` except for **how tiles are created**.

Instead of clustering tiles at the AEZ level, this script **divides a given ROI into Web-Mercator–aligned tiles** directly.  
The rest of the pipeline — downloading imagery, running models, segmentation, post-processing, and export — remains unchanged.

---

## Method: get_points(roi, directory)

### Purpose
Partition a Region of Interest (ROI) into a regular grid of **Web-Mercator–aligned tiles (zoom=17)** and persist the grid to disk as `status.csv`.  
Each row represents one tile intersecting the ROI and includes downstream processing flags.

In short:  
> It snaps the ROI to the standard tile lattice, filters to intersecting tiles, and writes a resumable job list.

---

### Signature
```python
def get_points(roi, directory) -> pandas.DataFrame
```

---

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `roi` | `ee.Geometry` / `ee.FeatureCollection` | Region of Interest defining the area to tile. |
| `directory` | `str` | Folder path where `status.csv` will be created or reused. |

---

### Returns
A **pandas.DataFrame** with one row per intersecting tile.

| Column | Type | Description |
|---------|------|-------------|
| `index` | int | Sequential tile ID (0-based) |
| `points` | tuple | ((top_left_lat, top_left_lon), (bottom_right_lat, bottom_right_lon)) |
| `overall_status` | bool | True if all stages complete |
| `download_status` | bool | Imagery download done |
| `model_status` | bool | Model inference done |
| `segmentation_status` | bool | Segmentation done |
| `postprocessing_status` | bool | Vectorization done |
| `plantation_status` | bool | Plantation mask done |

If `{directory}/status.csv` already exists, it is read and reused (idempotent).

---

### What It Creates on Disk

**File:** `{directory}/status.csv`

Each row stores:
- Tile coordinates
- Stage-by-stage flags (initialized to `False`)

Per-tile folders (`{directory}/0`, `{directory}/1`, etc.) are created **later** by the runner, not by `get_points()`.

---

## Tiling Scheme & Geometry Math

### Coordinate Space

| Property | Value |
|-----------|--------|
| Projection | Web Mercator |
| Zoom Level | 17 |
| Tile Block | 16×16 standard tiles |
| Base Tile Size | 256×256 px |
| Effective Tile Size | 4096×4096 px |
| Scale Multiplier | 16 |
| Ground Resolution | ≈1.19 m/pixel |

Each tile corresponds to a 16×16 group of standard web tiles, later subdivided into 256×256 patches for model inference.

---

### Steps

1. **Get ROI bounds**
   ```python
   bounds = roi.bounds().coordinates().get(0).getInfo()
   ```
   Extracts min/max latitudes and longitudes.

2. **Snap ROI to Web-Mercator lattice**
   ```python
   tile_x, tile_y = latlon_to_tile_xy(lats[-1], lons[0], zoom=17)
   top_left_lat, top_left_lon = tile_xy_to_latlon(tile_x, tile_y, zoom=17)
   ```
   Anchors grid to top-left tile corner at zoom 17.

3. **Compute grid coverage**
   Converts ROI span to pixel coordinates, divides by `(256 * 16)` to determine grid extent:
   ```python
   iterations = ceil(max(|dx|, |dy|) / (256 * 16))
   ```

4. **Generate tile anchors**
   Builds all grid anchors via `get_n_boxes()` (lat/lon product of corners).

5. **Form rectangles and filter intersections**
   For each tile:
   ```python
   rectangle = ee.Geometry.Rectangle([
       (top_left_lon, top_left_lat),
       (bottom_right_lon, bottom_right_lat)
   ])
   ```
   Keeps only those tiles intersecting the ROI with `ee.ErrorMargin(1)`.

6. **Write status.csv**
   Initializes all status flags to `False`.

---

## CSV Schema

**File:** `{directory}/status.csv`

| Column | Type | Meaning |
|---------|------|----------|
| `index` | int | Tile ID (used for folder naming) |
| `points` | tuple | ((lat1, lon1), (lat2, lon2)) |
| `overall_status` | bool | Entire tile pipeline complete |
| `download_status` | bool | Imagery + chunking done |
| `model_status` | bool | Model inference done |
| `segmentation_status` | bool | Segmentation done |
| `postprocessing_status` | bool | Feature extraction done |
| `plantation_status` | bool | YOLO plantation mask done |

All flags = `False` initially; updated by the main runner.

---

## How the CSV Is Used

The pipeline’s `run(...)` function:
1. Reads tiles where `overall_status == False`
2. For each tile:
   - Creates `{directory}/{index}/`
   - Downloads imagery → `{index}/field.tif`
   - Chunks into `{index}/chunks/`
   - Runs inference, segmentation, postprocessing, plantation steps
   - Updates status flags per stage
3. Marks `overall_status = True` on completion.

Because `get_points` is **idempotent**, you can resume partially completed runs safely.

---

## Directory Structure (End-to-End)

```
{directory}/
├─ status.csv                       # Created by get_points
├─ output.log                       # Pipeline log
├─ 0/
│  ├─ field.tif
│  ├─ chunks/
│  │  ├─ chunk_0_0.tif
│  ├─ logits_bounds.pickle
│  ├─ instance_predicted.pickle
│  ├─ all.*
│  ├─ plantation.*
├─ 1/
│  └─ ...
├─ ...
├─ all_done
├─ all.shp / plantation.shp
└─ {directory}_boundaries_*.shp
```

---

## Behavior Details & Edge Cases

| Behavior | Description |
|-----------|-------------|
| **Snapping & coverage** | Tiles are aligned to Web-Mercator grid; full bounding box covered. Non-intersecting tiles discarded. |
| **Tile extent** | Each tile = 4096×4096 px (256×16) at zoom 17. |
| **Resumability** | Existing CSV is reused; delete to regenerate. |
| **Error margin** | Uses `ee.ErrorMargin(1)` (~1 m). Adjust for very narrow ROIs. |
| **Indexing** | Row index = tile folder name. Do not reorder manually. |

---

**In summary:**  
`get_points()` is the grid generator for block-level runs. It aligns tiles neatly on Web-Mercator boundaries, creates a resumable job ledger (`status.csv`), and allows the rest of the pipeline to process tiles independently and safely resume unfinished tasks.
