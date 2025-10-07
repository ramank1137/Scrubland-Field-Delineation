#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List class names from a categorical GeoTIFF as <level_1>_<level_2>_<level_3>.
Robust against: multi-band RAT, Min/Max (range) RAT, missing Value column,
aux.xml RAT, ESRI VAT .dbf, band metadata, float codes, and palette index mapping.

Usage:
  python lulc_list_classes.py raster.tif --out_csv classes.csv --debug
"""

import os, re, sys, csv, math, json, xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from osgeo import gdal, ogr

EPS = 1e-6

def dprint(debug, *a):
    if debug: print(*a, file=sys.stderr)

def slug(s: str) -> str:
    return re.sub(r"\s+", "_", (s or "").strip())

def nkey(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", (s or "").lower().replace(" ", "_"))

def split3(label: str):
    parts = [p.strip() for p in re.split(r"[>/\|:_-]+", label or "") if p.strip()]
    if len(parts) == 0: return ("","","")
    if len(parts) == 1: return ("","",parts[0])
    if len(parts) == 2: return (parts[0],parts[1],"")
    return (parts[0],parts[1],parts[2])

def get_unique_values(ds, b, debug=False):
    band = ds.GetRasterBand(b)
    bx, by = band.GetBlockSize()
    if not bx or not by: bx, by = 256, 256
    xsize, ysize = band.XSize, band.YSize
    nod = band.GetNoDataValue()
    vals = set()
    try:
        import numpy as np
        for y in range(0, ysize, by):
            rows = min(by, ysize-y)
            for x in range(0, xsize, bx):
                cols = min(bx, xsize-x)
                arr = band.ReadAsArray(x, y, cols, rows)
                if arr is None: continue
                arr = arr.astype("float64", copy=False)
                if nod is not None:
                    arr = arr[arr != float(nod)]
                vals.update(np.unique(arr).tolist())
    except Exception:
        for y in range(0, ysize, by):
            rows = min(by, ysize-y)
            for x in range(0, xsize, bx):
                cols = min(bx, xsize-x)
                arr = band.ReadAsArray(x, y, cols, rows)
                if arr is None: continue
                for row in (arr if isinstance(arr, list) else [arr]):
                    for v in (row if isinstance(row, (list, tuple)) else [row]):
                        try:
                            fv = float(v)
                        except: continue
                        if nod is not None and abs(fv - float(nod)) < EPS: continue
                        vals.add(fv)
    out = sorted(vals)
    # collapse near-integers to their int for prettier matching
    pretty = [int(round(v)) if abs(v-round(v))<EPS else v for v in out]
    if debug: dprint(True, f"[Band {b}] unique values count: {len(pretty)} (showing up to 20): {pretty[:20]}")
    return pretty

# ---------- RAT via GDAL API ----------
def read_rat_gdal(band, debug=False):
    rat = band.GetDefaultRAT()
    if rat is None:
        return None
    cols = []
    for i in range(rat.GetColumnCount()):
        cols.append({
            "idx": i,
            "name": rat.GetNameOfCol(i) or f"col_{i}",
            "type": rat.GetTypeOfCol(i),   # 0=int,1=real,2=string
            "usage": rat.GetUsageOfCol(i)  # enums in gdal
        })
    if debug:
        dprint(True, "  RAT columns:",
               ", ".join([f"{c['name']}[t{c['type']} u{c['usage']} i{c['idx']}]" for c in cols]))

    # identify value/min/max by usage first
    value_idx = next((c["idx"] for c in cols if c["usage"]==gdal.GFU_Generic and nkey(c["name"]) in
                      ("value","classvalue","class_id","classid","code","id","pixelvalue")), None)
    min_idx   = next((c["idx"] for c in cols if c["usage"]==gdal.GFU_Min), None)
    max_idx   = next((c["idx"] for c in cols if c["usage"]==gdal.GFU_Max), None)

    # pick string columns that likely carry names
    str_cols = [c for c in cols if c["type"]==gdal.GFT_String]
    # prefer columns with 'level' or 'name' or 'class' in them
    prefer = [c for c in str_cols if re.search(r"(level|name|class|label)", c["name"], re.I)]
    string_order = prefer + [c for c in str_cols if c not in prefer]

    rows = []
    for r in range(rat.GetRowCount()):
        rec = {"value": None, "min": None, "max": None, "names": []}
        if value_idx is not None:
            try: rec["value"] = float(rat.GetValueAsString(r, value_idx))
            except: pass
        if min_idx is not None:
            try: rec["min"] = float(rat.GetValueAsString(r, min_idx))
            except: pass
        if max_idx is not None:
            try: rec["max"] = float(rat.GetValueAsString(r, max_idx))
            except: pass
        for c in string_order:
            s = rat.GetValueAsString(r, c["idx"])
            if s and s.strip(): rec["names"].append(s.strip())
        rows.append(rec)

    # if no value/min/max at all, assume row index == palette index
    if all(rec["value"] is None and rec["min"] is None and rec["max"] is None for rec in rows):
        for i,rec in enumerate(rows): rec["value"] = float(i)

    if debug:
        dprint(True, f"  RAT rows: {len(rows)}; sample:",
               rows[0] if rows else "none")

    return rows

# ---------- .aux.xml RAT ----------
def read_aux_xml(path, debug=False):
    cands = [path+".aux.xml", os.path.splitext(path)[0]+".aux.xml"]
    for aux in cands:
        if not os.path.exists(aux): continue
        try: tree = ET.parse(aux); root = tree.getroot()
        except Exception: continue
        rat = root.find(".//GDALRasterAttributeTable")
        if rat is None: continue
        # FieldDefn with Usage and Name
        defs = []
        for fd in rat.findall(".//FieldDefn"):
            defs.append({
                "Name": fd.findtext("Name") or "",
                "Usage": fd.findtext("Usage") or "",
                "Type": fd.findtext("Type") or "",
            })
        rows = []
        for row in rat.findall(".//Row"):
            Fs = row.findall("F")
            if not Fs: Fs = row.findall("V")
            vals = [(f.text or "") for f in Fs]
            rows.append(vals)
        if defs and rows:
            if debug:
                dprint(True, f"  AUX XML found: {aux} with {len(defs)} fields, {len(rows)} rows")
            # map to dicts like in RAT
            out = []
            for r in rows:
                d = {}
                for i,fd in enumerate(defs):
                    nm = fd["Name"] or f"col_{i}"
                    d[nm] = r[i] if i < len(r) else ""
                out.append(d)
            return defs, out
    return None

# ---------- ESRI VAT DBF ----------
def read_vat_dbf(path, debug=False):
    folder, name = os.path.split(path)
    stem = os.path.splitext(name)[0]
    cands = [f"{stem}.vat.dbf", f"{stem}.VAT.DBF", "VAT.DBF", f"{stem}.DBF"]
    for c in cands:
        p = os.path.join(folder, c)
        if not os.path.exists(p): continue
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if not drv: continue
        ds = drv.Open(p, 0)
        if not ds: continue
        lyr = ds.GetLayer(0)
        defn = lyr.GetLayerDefn()
        fields = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
        rows = []
        for f in lyr:
            rows.append({fld: f.GetField(fld) for fld in fields})
        ds = None
        if rows:
            if debug: dprint(True, f"  VAT DBF found: {p} with {len(rows)} rows")
            return fields, rows
    return None

# ---------- band metadata ----------
def read_band_metadata(band, debug=False):
    mapping = defaultdict(lambda: ["","",""])
    doms = list(band.GetMetadataDomainList() or []) + [""]
    for d in doms:
        md = band.GetMetadata(d)
        if not md: continue
        for k,v in md.items():
            k0 = nkey(k)
            m = re.match(r"^(level)([123])_(\d+)$", k0)               # level1_23
            if m:
                idx = int(m.group(2))-1; val = int(m.group(3))
                mapping[val][idx] = v; continue
            m = re.match(r"^value_(\d+)_(level)([123])$", k0)         # value_23_level3
            if m:
                val = int(m.group(1)); idx = int(m.group(3))-1
                mapping[val][idx] = v; continue
            m = re.match(r"^(label|name)_(\d+)$", k0)                 # label_23
            if m:
                val = int(m.group(2)); mapping[val] = list(split3(v)); continue
    cleaned = {val: tuple(parts) for val,parts in mapping.items() if any(parts)}
    if debug and cleaned: dprint(True, f"  Band metadata per-value entries: {len(cleaned)}")
    return cleaned if cleaned else None

def choose_l123(names):
    # pick up to 3 most relevant strings for name construction
    pri = sorted(names, key=lambda s: (0 if re.search(r"(level|name|class|label)", s, re.I) else 1, len(s)))
    parts = [t for t in pri if t.strip()][:3]
    if not parts: return ("","","")
    if len(parts)==1: return split3(parts[0])
    if len(parts)==2: return (parts[0], parts[1], "")
    return (parts[0], parts[1], parts[2])

def build_mapping_for_band(ds, b, tif_path, debug=False):
    band = ds.GetRasterBand(b)
    direct = {}           # value(float/int) -> (l1,l2,l3,class_name)
    ranges = []           # (min,max,(l1,l2,l3,class_name))
    index_rows = {}       # row_index -> (l1,l2,l3,class_name) when RAT lacks value/min/max

    # 1) RAT via GDAL
    rat = read_rat_gdal(band, debug=debug)
    if rat:
        for i,rec in enumerate(rat):
            if rec["names"]:
                l1,l2,l3 = None,None,None
                # try to detect explicit level columns by text tokens
                l1,l2,l3 = choose_l123(rec["names"])
                cname = "_".join(filter(None,[slug(l1),slug(l2),slug(l3)])).strip("_")
                if rec["value"] is not None:
                    direct[rec["value"]] = (l1,l2,l3,cname)
                elif rec["min"] is not None and rec["max"] is not None:
                    ranges.append((rec["min"], rec["max"], (l1,l2,l3,cname)))
                else:
                    index_rows[float(i)] = (l1,l2,l3,cname)
        if debug:
            dprint(True, f"  RAT map sizes: direct={len(direct)}, ranges={len(ranges)}, indexRows={len(index_rows)}")

    # 2) .aux.xml RAT
    if not direct and not ranges and not index_rows:
        aux = read_aux_xml(tif_path, debug=debug)
        if aux:
            defs, rows = aux
            norm = {nkey(d["Name"]): d for d in defs}
            vfield = None
            for k,d in norm.items():
                if d.get("Usage","")== "GFU_Min": vmin = d["Name"]
                if d.get("Usage","")== "GFU_Max": vmax = d["Name"]
            # try value by name
            for cand in ("value","classvalue","class_id","classid","code","id","pixelvalue"):
                if cand in norm: vfield = norm[cand]["Name"]; break
            string_names = [d["Name"] for d in defs if (d.get("Type","") == "GFT_String" or re.search(r"(name|class|level|label)", d.get("Name",""), re.I))]
            for row in rows:
                names = [row.get(n,"") for n in string_names if row.get(n,"").strip()]
                l1,l2,l3 = choose_l123(names)
                cname = "_".join(filter(None,[slug(l1),slug(l2),slug(l3)])).strip("_")
                if vfield and row.get(vfield,"")!="":
                    try: direct[float(row[vfield])] = (l1,l2,l3,cname)
                    except: pass
                elif "vmin" in locals() and "vmax" in locals() and row.get(vmin,"")!="" and row.get(vmax,"")!="":
                    try: ranges.append((float(row[vmin]), float(row[vmax]), (l1,l2,l3,cname)))
                    except: pass

    # 3) ESRI VAT .dbf
    if not direct and not ranges and not index_rows:
        vat = read_vat_dbf(tif_path, debug=debug)
        if vat:
            fields, rows = vat
            nm = {nkey(f): f for f in fields}
            vfield = nm.get("value") or nm.get("classvalue") or nm.get("code") or nm.get("id") or nm.get("pixelvalue")
            l1f = nm.get("level_1") or nm.get("level1") or nm.get("l1")
            l2f = nm.get("level_2") or nm.get("level2") or nm.get("l2")
            l3f = nm.get("level_3") or nm.get("level3") or nm.get("l3")
            lab = nm.get("label") or nm.get("name") or nm.get("class_name")
            for row in rows:
                if not vfield or row.get(vfield) in (None,""): continue
                try: v = float(row[vfield])
                except: continue
                if l1f and l2f and l3f:
                    l1,l2,l3 = row.get(l1f,""), row.get(l2f,""), row.get(l3f,"")
                else:
                    l1,l2,l3 = split3(str(row.get(lab,"")))
                cname = "_".join(filter(None,[slug(l1),slug(l2),slug(l3)])).strip("_")
                direct[v] = (l1,l2,l3,cname)

    # 4) band metadata per-value
    if not direct and not ranges and not index_rows:
        md = read_band_metadata(band, debug=debug)
        if md:
            for v,parts in md.items():
                l1,l2,l3 = parts
                cname = "_".join(filter(None,[slug(l1),slug(l2),slug(l3)])).strip("_")
                direct[float(v)] = (l1,l2,l3,cname)

    return {"direct": direct, "ranges": ranges, "index": index_rows}

def match_name(v, mapping):
    # Try exact float/int key match with tolerance
    for key in mapping["direct"].keys():
        if (isinstance(key, float) or isinstance(v, float)):
            if abs(float(key) - float(v)) < EPS:
                return mapping["direct"][key]
        else:
            if key == v: return mapping["direct"][key]
    # Try ranges
    for mn,mx,rec in mapping["ranges"]:
        if mn is None or mx is None: continue
        if mn - EPS <= float(v) <= mx + EPS:
            return rec
    # Try index rows (palette index): cast to nearest int
    vi = int(round(float(v)))
    if float(v).is_integer() and float(vi) == float(v) and vi in mapping["index"]:
        return mapping["index"][vi]
    return None

def main():
    import argparse
    p = argparse.ArgumentParser(description="Extract <level_1>_<level_2>_<level_3> class names from GeoTIFF.")
    p.add_argument("tif")
    p.add_argument("--out_csv", help="Write value,level_1,level_2,level_3,class_name")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    ds = gdal.Open(args.tif, gdal.GA_ReadOnly)
    if ds is None:
        print(f"ERROR: cannot open {args.tif}", file=sys.stderr); sys.exit(2)

    best_rows, best_band, best_resolved = None, None, -1
    for b in range(1, ds.RasterCount+1):
        dprint(args.debug, f"Scanning band {b} …")
        mapping = build_mapping_for_band(ds, b, args.tif, debug=args.debug)
        vals = get_unique_values(ds, b, debug=args.debug)
        rows = []
        resolved = 0
        for v in vals:
            rec = match_name(v, mapping)
            if rec:
                l1,l2,l3,cname = rec
                resolved += 1
            else:
                l1,l2,l3 = ("L1_UNKNOWN","L2_UNKNOWN",f"value_{int(v) if float(v).is_integer() else v}")
                cname = "_".join([slug(l1),slug(l2),slug(l3)])
            rows.append({"value": v, "level_1": l1, "level_2": l2, "level_3": l3, "class_name": cname})
        if args.debug:
            dprint(True, f"[Band {b}] resolved {resolved}/{len(vals)}")
        if resolved > best_resolved:
            best_rows, best_band, best_resolved = rows, b, resolved

    # Print names
    for r in best_rows:
        print(r["class_name"])

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["band","value","level_1","level_2","level_3","class_name"])
            w.writeheader()
            for r in best_rows:
                w.writerow({"band": best_band, **r})
        print(f"\nWrote {len(best_rows)} rows → {os.path.abspath(args.out_csv)}")

if __name__ == "__main__":
    main()
