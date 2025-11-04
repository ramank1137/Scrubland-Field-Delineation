# Implementation Details: GEE Script 3 — Temporal Multi-Year Sampling

**Script name:** `3_store_embeddings_run_lulcv4 (temporal multi year sampling).py`

---

<details>
<summary><strong>📑 Table of Contents</strong></summary>

- [Overview](#overview)
- [High-level Flow](#high-level-flow)
- [Class Schema (IndiaSAT v4)](#class-schema-indiasat-v4)
- [Objects & Simple Setup](#objects--simple-setup)
- [Export Utilities](#export-utilities)
- [Sampling Strategy](#sampling-strategy)
- [Embeddings Preparation](#embeddings-preparation)
- [Label Source and Stratified Subset](#label-source-and-stratified-subset)
- [Batch Exports of Sampled Embeddings](#batch-exports-of-sampled-embeddings)
- [Aggregating Sample Tables & Training a Classifier](#aggregating-sample-tables--training-a-classifier)
- [Date Helper](#date-helper)
- [Main Orchestrator for a Given ROI Part](#main-orchestrator-for-a-given-roi-part)
- [AOI Splitting & Invocation](#aoi-splitting--invocation)
- [Practical Tips & Checks](#practical-tips--checks)

</details>

---

## Overview

This module generates **annual IndiaSAT-compatible LULC maps at 10 m** resolution for each Agro-Ecological Zone (AEZ).  
It integrates Google Annual Satellite Embeddings (64-D) with multi-year sampling, Random Forest classification, and crop-intensity inference to build temporally consistent LULC layers (2017–2025).

---

## High-level Flow

For each `AEZ_no ∈ {1…20}`:
1. Build annual windows `[YYYY-07-01, (YYYY+1)-06-30]` (2017→2025).
2. Generate base masks:
   - Built-up (DynamicWorld + S2 NDWI/NDVI cleaning)
   - Water (S1 Kharif + DW Rabi/Zaid + S2 NDWI cleaning)
   - Barren (DynamicWorld bare)
3. Classify **Farm / Non-Agro / Plantation** using embeddings + AEZ-local RF model.
4. Subdivide farms by crop intensity and Non-Agro by tree vs scrubland.
5. Combine background and embedding layers; export final LULC maps.

---

## Class Schema (IndiaSAT v4)

| Concept | Code | Notes |
|----------|------|--------|
| Background | 0 | Default base |
| Built-up | 1 | DynamicWorld + S2 |
| Water (Kharif/Rabi/Zaid) | 2 / 3 / 4 | Seasonal |
| Barren | 7 | From DW bare |
| Farm (pre-split) | 5 | Intermediate |
| Single Kharif | 8 | Final |
| Single Non-Kharif | 9 | Final |
| Double crop | 10 | Final |
| Triple crop | 11 | Final |
| Scrubland | 12 | Final |
| Plantation | 13 | Final |

---

## Objects & Simple Setup

```python
roi_boundary = ee.FeatureCollection("users/mtpictd/agro_eco_regions").filter(ee.Filter.eq("ae_regcode", AEZ_no))
filename_prefix = "AEZ_" + str(AEZ_no)
mapping = {"farm": 1, "plantation": 2, "scrubland": 3, "rest": 0}
```

- **roi_boundary:** Loads the AEZ region polygon.  
- **filename_prefix:** Used in naming outputs.  
- **mapping:** Links intermediate classifier IDs to downstream class codes.

---

## Export Utilities

### `write_to_gee(fc, asset_name)`
Exports a FeatureCollection to a GEE asset while polling until completion.  
Useful for safe, synchronous exports.

### `export_samples_in_chunks(image, samples, chunk_size, asset_prefix)`
Exports sampled embeddings in multiple batches to avoid quota/timeouts.  
Returns a list of asset IDs for subsequent training.

---

## Sampling Strategy

`stratified_sample_by_tile_label(...)`
- Draws samples across (tile, label) strata to ensure balanced spatial and class representation.
- Supports `fraction`, `seed`, and `min_per_group` for deterministic subsampling.
- Ensures total dataset size < 100 MB by limiting fraction (e.g., 0.33).

---

## Embeddings Preparation

```python
emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
emb_2025 = emb.filterDate('2024-01-01', '2025-01-01').filterBounds(roi_boundary).mosaic()
```

- Mosaics annual embeddings into continuous images.
- Allows temporal sampling (e.g., 2023–2025).
- Each mosaic clipped to AEZ extent.

---

## Label Source and Stratified Subset

```python
samples = ee.FeatureCollection("projects/raman-461708/assets/gee_samples_all").filter(ee.Filter.eq('aez_no', AEZ_no))
samples = stratified_sample_by_tile_label(samples, fraction=0.33, seed=42, min_per_group=50)
```

- Selects a subset of labeled data per AEZ.
- Maintains per-(tile, label) balance.
- Reduces total memory for 3-year embeddings.

---

## Batch Exports of Sampled Embeddings

```python
sample_assets_2025 = export_samples_in_chunks(emb_2025, samples, chunk_size=30000, asset_prefix=f'projects/raman-461708/assets/AEZ_{AEZ_no}_samples_from_local_2025')
sample_assets = sample_assets_2025 + sample_assets_2024 + sample_assets_2023
```

- Each year’s embeddings are sampled and exported in parallel batches.  
- Exported tables contain sampled pixel embeddings with attached labels.

---

## Aggregating Sample Tables & Training a Classifier

### `get_samples(roi)`
- Merges per-chunk tables across years into one FeatureCollection.
- Filters by AOI geometry.

### `get_classifier(bandnames, roi)`
- Trains a Random Forest (100 trees, seed=42) with `classProperty='label'`.
- Uses 64-D embedding bands as predictors.

---

## Date Helper

`get_emb_date(date: str) -> str`
- Converts `'YYYY-07-01'` → `'YYYY+1-01-01'`.  
- Aligns July–June agri years with Jan–Jan embedding datasets.

---

## Main Orchestrator for a Given ROI Part

### `get_lulc(roi_boundary, parts="")`

**Pipeline:**
1. Iterate over yearly windows (2017–2025).  
2. Build base masks → combine by precedence.  
3. Compute `cropping_frequency_img` for farms.  
4. Align embeddings via `get_emb_date()`; train classifier.  
5. Classify embeddings → remap: `{farm:5, scrub:12, plantation:13}`.  
6. Integrate with base layers.  
7. Apply cropping intensity (clusters 8–11).  
8. Export to GEE (`pyramidingPolicy='mode'`).

**Outputs:**  
`projects/raman-461708/assets/AEZ_<no>_v4_temporal_3years_<parts>_<YYYY-07-01>_<YYYY+1-06-30>`

---

## AOI Splitting & Invocation

```python
if AEZ_no in [6,]:
    parts_fc = split_roi(roi_boundary)
    part1 = parts_fc.filter(ee.Filter.eq('part', 1))
    part2 = parts_fc.filter(ee.Filter.eq('part', 2))
    get_lulc(part1, "_part1")
    get_lulc(part2, "_part2")
else:
    get_lulc(roi_boundary, "")
```

- Splits large AOIs (e.g., AEZ 6) into halves to prevent memory overflow.  
- Others processed as full AEZs.

---

## Practical Tips & Checks

- Ensure GEE asset write access for `projects/raman-461708/assets/...`.  
- Serial `write_to_gee` is safer; use parallel only for exports.  
- Always dynamically fetch bandnames:
  ```python
  bandnames = embeddings.bandNames()
  get_classifier(bandnames, roi_boundary)
  ```
- AEZ runs can be parallelized, but respect Earth Engine’s concurrent task limits (usually ≤300).

