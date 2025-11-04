# Implementation Details: Local Compute Script 2 — Boundary Refinement & Stratified Sampling

**Script name:** `2_sampling_locally_workers.py`

---

<details>
<summary><strong>📑 Table of Contents</strong></summary>

- [Overview](#overview)
- [Output File Structure](#output-file-structure)
- [Labels](#labels)
- [Inputs](#inputs)
- [Overview of Operations](#overview-of-operations)
- [The Rules (As Implemented)](#the-rules-as-implemented)
- [Configuration Knobs](#configuration-knobs)
- [Basic Usage](#basic-usage)
- [Expected Folders and Files](#expected-folders-and-files)
- [Method-by-Method Documentation](#method-by-method-documentation)
  - [Logging & Multiprocessing](#logging--multiprocessing)
  - [Discovery, I/O, CRS, and Small Helpers](#discovery-io-crs-and-small-helpers)
  - [Geometry Hygiene & Set Operations](#geometry-hygiene--set-operations)
  - [Rule Filters & Precedence](#rule-filters--precedence)
  - [Sampling & CSVs](#sampling--csvs)
  - [Tile & AEZ Drivers](#tile--aez-drivers)
- [Notes](#notes)

</details>

---

## Overview

This script takes boundaries detected inside each AEZ, applies the rule filters for **farms**, **plantations**, and **non-agri/scrub**, resolves overlaps by priority (**Plantation > Farm > Scrubland**), and draws random and uniform training points per class.

Outputs are small CSVs per tile and one collated CSV for import into GEE.

It is optimized for **local compute** using GeoPandas/Shapely (vectors), Rasterio (raster checks), and Python multiprocessing.

---

## Output File Structure

**Per-tile CSVs:**  
`AEZ_<no>/samples/tile_<i>.csv` with columns:
```
lat, lon, label, tile_index, aez_no
```

**Combined CSV for GEE:**  
`export/gee_samples_all.csv` with columns:
```
latitude, longitude, label, tile_index, aez_no
```

---

## Labels

| Label | Class |
|--------|--------|
| 1 | Farm |
| 2 | Plantation |
| 3 | Scrubland / Non-agri |

---

## Inputs

### AEZ Boundaries (Vectors)
Files like `AEZ_<aez>_boundaries_0.shp`, `..._1.gpkg`, etc., under `BOUNDARY_SEARCH_ROOT`.

Required columns:
- `rect` (rectangularity, float)
- `ent` (entropy, float)
- `size` (area in m², float)
- `class` (string; “plantation” matched case-insensitively)

### LULC Raster (GeoTIFF)
`LULC_RASTER_PATH` — single-band categorical map.  
Used to ensure scrubland polygons aren’t mostly cropland.

### Tiles List
`AEZ_<aez>/status.csv` with a `points` column:
```
((lon1, lat1), (lon2, lat2))
```
If using `(lat, lon)` order, set:
```python
POINTS_ARE_LON_LAT = False
```

---

## Overview of Operations

1. Load boundaries for one AEZ, repair invalid geometries, harmonize CRS.  
2. Apply rules per tile:

   - **Farms:** `ent < 1.0`, `rect ≥ 0.67`, `500 < size < 2000 m²`, clustered ≥3 after 10 m buffer.  
   - **Scrubland/Non-agri:** `60000 ≤ size ≤ 5,000,000 m²` and `< 50%` cropland pixels.  
   - **Plantations:** `class == "plantation"` and `area ∈ [1,000, 20,000] m²`.

3. Apply −5 m tile buffer to avoid edge artifacts.  
4. Resolve overlaps (**Plantation > Farm > Scrubland**).  
5. Sample ~150 points per class per tile (UTM-based, reprojected to WGS84).  
6. Write CSVs per tile and collated CSV for GEE.

---

## The Rules (As Implemented)

| Class | Filters (all must pass) |
|--------|--------------------------|
| **Farm (1)** | `ent < 1.0`, `rect ≥ 0.67`, `500 < size < 2000 m²`, clusters ≥3 after 10 m buffer |
| **Scrub (3)** | `60000 ≤ size ≤ 5,000,000 m²`, `< 50%` pixels in {8,9,10,11} |
| **Plantation (2)** | `class == "plantation"`, area ∈ [1,000, 20,000] m² (UTM) |

**Overlap priority:** `Plantation > Farm > Scrubland`

---

## Configuration Knobs

| Parameter | Description |
|------------|--------------|
| `BOUNDARY_SEARCH_ROOT` | Directory where AEZ boundaries live |
| `VECTOR_EXTS` | Allowed vector formats |
| `LULC_RASTER_PATH` | GeoTIFF path for scrubland cropland check |
| `CLASSES_OF_INTEREST` | `{8,9,10,11}` cropland classes for 50% rule |
| `CLASS_TARGETS` | `{1:150, 2:150, 3:150}` samples per class per tile |
| `FARM_CLUSTER_BUFFER_M` | 10.0 (farm clustering buffer) |
| `TILE_INNER_BUFFER_M` | 5.0 (shrink tile before clipping) |
| `POINTS_ARE_LON_LAT` | Set False if CSV uses (lat, lon) |

---

## Basic Usage

### 1. Generate samples for an AEZ (parallel)
```python
sample_points_from_tiles_parallel(8)  # Example: AEZ 8
```
Reads `AEZ_8/status.csv`, processes each tile, and writes:
```
AEZ_8/samples/tile_*.csv
```

### 2. Collate samples for multiple AEZs
```python
collate_samples_to_gee_csv([6,7,8,9], Path("export/gee_samples_all.csv"))
```

Produces `export/gee_samples_all.csv` with headings:
```
latitude, longitude, label, tile_index, aez_no
```

> In `__main__`, the sampling loop is commented; the collate call is active. Uncomment to run both steps.

---

## Expected Folders and Files

```
BOUNDARY_SEARCH_ROOT/
  AEZ_8_boundaries_0.shp (or .gpkg/.geojson/.parquet/.feather)
  AEZ_8_boundaries_1.shp
data/
  pan_india_lulc_v3_2023_2024.tif
AEZ_8/
  status.csv
  samples/
    tile_0.csv
    tile_1.csv
export/
  gee_samples_all.csv
```

---

## Method-by-Method Documentation

### Logging & Multiprocessing

**`log(msg: str)`**  
Prints timestamp + PID. Used to track progress across workers.

**`_dynamic_workers_for_aez(aez_no: int) -> int`**  
Determines worker count based on CPU and boundary shards.  
Scales down on macOS/Windows to prevent memory overload.

**`_worker_init(aez_no: int)`**  
Initializes worker: loads and caches boundaries per AEZ.

**`_clear_parent_cache()`**  
Clears parent boundary cache after AEZ completes.

---

### Discovery, I/O, CRS, and Small Helpers

**`_find_boundary_files(aez_no: int) -> list[str]`**  
Globs boundary shards under `BOUNDARY_SEARCH_ROOT`.

**`load_aez_boundaries(aez_no: int) -> gpd.GeoDataFrame`**  
Reads, repairs, harmonizes CRS, filters polygonal types, merges into GeoDataFrame.

**`ensure_columns(df, cols)`**  
Adds missing columns with NaN defaults.

**`utm_crs_for_lonlat(lon, lat) -> CRS`**  
Determines appropriate UTM CRS.

**`to_utm(gdf, lon, lat) -> (gdf_utm, crs)`**  
Reprojects GeoDataFrame to local UTM CRS.

**`parse_points_to_tile(points_str, assume_lonlat=True) -> Polygon`**  
Parses `status.csv` “points” into rectangular tile polygon.

**`clip_gdf_to_tile(gdf, tile)`**  
Spatially clips features to tile extent.

**`negative_buffer_polygon(poly, meters, utm_crs, src_crs)`**  
Applies negative buffer (−5 m) to prevent edge artifacts.

---

### Geometry Hygiene & Set Operations

**`_safe_fix_invalid(gdf)`**  
Repairs invalid polygons (`buffer(0)`); drops non-polygon geometries.

**`_fix_polygonal_geom(g)`**  
Converts GeometryCollection → Polygon/MultiPolygon; repairs self-intersections.

**`dissolve_all(gdf)`**  
Merges polygons, explodes multiparts, repairs geometry, removes slivers.

**`difference_all(a, b)`**  
Computes `a − b` safely; ensures exclusivity between class layers.

---

### Rule Filters & Precedence

**`percent_interest_in_polygon(poly, raster_path, classes_of_interest)`**  
Computes % of pixels in polygon belonging to `classes_of_interest`.  
Used to exclude cropland-like scrub areas.

**`cluster_farms(farm_gdf, utm_crs)`**  
Buffers by 10 m, unions clusters, counts intersections ≥3, retains clustered farms.

**`prepare_class_geometries(all_gdf, tile_poly, lulc_path, src_crs)`**  
Core of refinement logic:
1. Clip boundaries to tile.
2. Ensure required columns.
3. Compute true area for plantation filter.
4. Apply rules:
   - **Farm:** entropy, rectangularity, size, cluster ≥3.  
   - **Scrub:** area + <50% cropland.  
   - **Plantation:** class filter + true area.
5. Apply −5 m buffer.
6. Dissolve each class.
7. Enforce precedence → `{1: farm, 2: plantation, 3: scrub}`.

---

### Sampling & CSVs

**`clean_for_sampling(gdf, to_crs)`**  
Fixes geometry, reprojects, drops invalid/micro polygons.

**`weighted_random_points_in_gdf(gdf_utm, n, rng)`**  
Generates random, area-weighted points within polygons.

**`stratified_sample_points(class_geoms, tile_poly, aez_no, tile_index, seed=...)`**  
Draws stratified samples across {1,2,3}, converts back to WGS84, and tags AEZ/tile.

**`write_samples_csv(gdf_points, out_csv)`**  
Writes per-tile CSV (with header if empty).

**`collate_samples_to_gee_csv(aez_list, out_csv)`**  
Aggregates multiple AEZ sample CSVs into one GEE-compatible CSV.

---

### Tile & AEZ Drivers

**`process_tile(aez_no, tile_poly, tile_index)`**  
Processes a single tile → returns per-tile sample summary.

**`_process_tile_worker(args)`**  
Parallel worker version of `process_tile`.

**`sample_points_from_tiles(aez_no)`**  
Serial AEZ sampling loop.

**`sample_points_from_tiles_parallel(aez_no)`**  
Parallel AEZ sampling loop:
- On Linux (fork): shared boundaries (fast).  
- On macOS/Windows (spawn): per-worker loading.

---

## Notes

- **Units:** all areas/buffers in meters (UTM).  
- **Geometry hygiene:** invalid polygons repaired; empties dropped.  
- **Deterministic sampling:** seed combines AEZ + tile index.  
- **Performance:**  
  - Linux uses `fork` (shared memory).  
  - macOS/Windows use `spawn` (safe but memory-heavy).  
  - Worker count auto-adjusts to prevent crashes.
