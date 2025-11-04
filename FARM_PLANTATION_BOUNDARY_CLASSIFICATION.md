# Documentation: Farm–Plantation Boundary Classification & Export

This script processes **block-level candidate boundaries** and a **LULC mosaic (predicted_label band)** to produce two cleaned, exportable datasets:
- **Farm Boundaries**
- **Plantation Boundaries**

Both are exported to:
- Google Earth Engine (GEE) Assets
- Google Drive (GeoJSON format)

---

<details>
<summary><strong> Table of Contents</strong></summary>

- [1. Inputs & Assumptions](#1-inputs--assumptions)
- [2. Script Workflow (High Level)](#2-script-workflow-high-level)
- [3. Thresholds & Rationale](#3-thresholds--rationale)
- [4. Edge Cases & Safeguards](#4-edge-cases--safeguards)
- [5. Performance Notes](#5-performance-notes)
- [6. Validation Checklist](#6-validation-checklist)
- [7. Common Pitfalls & Fixes](#7-common-pitfalls--fixes)

</details>

---

## 1. Inputs & Assumptions

### LULC Image (`mosaic_img`)
- Must include a **band** named `predicted_label` (integer class IDs).
- Class expectations:

| Class | IDs | Usage |
|--------|-----|--------|
| Farms | 8, 9, 10, 11 | Used |
| Scrub | 7, 12 | Declared (not used) |
| Forest | 6 | Declared (not used) |
| Plantation | 13 | Used |

### Region of Interest (`roi`)
- Used to clip the LULC mosaic for spatial focus.

### Candidate Boundaries (`boundaries`)
- A `FeatureCollection` of polygon geometries with a `class` property.
- Excludes features with `class == 'plantation'` (to avoid overlap with plantation detections).

---

## 2. Script Workflow (High Level)

### 1 Prepare the LULC Layer
```python
lulc = mosaic_img.select('predicted_label').clip(roi).rename('lulc').toInt()
```
- Extracts and clips the `predicted_label` band to ROI.

### 2 Filter Candidate Boundaries
- Removes features already labeled `'plantation'`:
  ```python
  boundaries = boundaries.filter(ee.Filter.neq('class', 'plantation'))
  ```
- These are redundant detections from the plantation model.

### 3 Compute Per-Boundary Pixel Statistics
- For each boundary:
  ```python
  hist = lulc.reduceRegion(
      reducer=ee.Reducer.frequencyHistogram(),
      geometry=feature.geometry(),
      scale=10
  )
  ```
- This produces counts of pixels per class ID (10 m resolution).

### 4 Calculate Percent Coverage
For each feature:
- `farm_pct = % pixels in [8,9,10,11]`
- `plantation_pct = % pixels in [13]`

### 5 Split into Two Outputs
- **Farm boundaries:** `farm_pct > 50`
- **Plantation boundaries:** `plantation_pct > 50`

### 6 Export Cleaned Outputs
Each output is slimmed to essential attributes and exported:
- **To GEE Asset** — for reuse and speed.
- **To Google Drive (GeoJSON)** — for GIS visualization.

---

## 3. Thresholds & Rationale

| Parameter | Value | Purpose |
|------------|--------|----------|
| Coverage Threshold | 50% | Ensures majority of pixels support label |
| Scale | 10 m | Balances detail and runtime |
| Rationale | Practical cutoff balancing accuracy and performance; can be tuned (e.g., 60–70%) based on regional validation |

> The 50% threshold works well for heterogeneous landscapes, giving clean separations between dominant classes.

---

## 4. Edge Cases & Safeguards

| Case | Handling |
|-------|-----------|
| Empty or tiny polygons | `total == 0` → assigns 0% coverage to avoid division errors |
| Mixed polygons | Excluded if no class > 50% |
| Pre-labeled plantations | Excluded at the start |
| Class ID mismatch | Warns if class not found in histogram |

---

## 5. Performance Notes

- `frequencyHistogram(scale=10)` is optimal for 10 m datasets.
- If you encounter limits:
  - Use smaller geometries or tile subsets.
  - Add `maxPixels` in reduceRegion.
- `.map(get_distribution)` enables efficient server-side iteration.
- Avoid per-feature client operations.

---

## 6. Validation Checklist

| Check | Method |
|--------|--------|
| Visual overlay | Compare exported polygons over LULC mosaic in GEE |
| Purity stats | Compute mean/median of `farm_pct` and `plantation_pct` |
| Ground-truth comparison | Overlay with known labeled polygons or field data |

> Validation ensures your exported polygons match real-world classes spatially and statistically.

---

## 7. Common Pitfalls & Fixes

| Issue | Cause | Fix |
|--------|--------|------|
| Missing `farm_pct` / `plantation_pct` | Property not set in `get_distribution()` | Verify property assignment |
| Drive export confusion | Duplicate filenames | Use unique `description` or `fileNamePrefix` per class |
| Histogram key mismatch | Class IDs as strings | Convert keys to `str()` before lookup |
| Empty outputs | Class mismatch or threshold too high | Check LULC IDs and reduce threshold |
| Slow export | Large polygons | Use tiling or smaller ROI subsets |

---

**Note:**  
GEE Asset exports complete significantly faster than Drive exports and are recommended for iterative workflows.

