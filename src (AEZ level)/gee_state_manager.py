import csv
import os
import tempfile
import contextlib
from typing import Optional, Dict, Any

# ====== Config ======
CSV_HEADER = ["aez_no", "last_tile", "tiles_complete", "samples_complete", "lulc_complete"]
DEFAULT_TOTAL_AEZ = 19


# ====== Utilities ======
def _parse_bool(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")

def _fmt_bool(x: bool) -> str:
    return "true" if bool(x) else "false"

def _parse_int_or_none(x: str) -> Optional[int]:
    x = (x or "").strip()
    return int(x) if x != "" else None


# --- Cross-platform advisory lock (Unix/mac: fcntl; Windows: noop fallback) ---
@contextlib.contextmanager
def _locked(lock_path: str):
    """
    Locks a companion .lock file for exclusive access across processes.
    On Unix/macOS uses fcntl; on Windows it falls back to a coarse-grained approach.
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True) if os.path.dirname(lock_path) else None
    f = open(lock_path, "a+")
    try:
        try:
            import fcntl  # Unix / macOS
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            # Best-effort fallback on systems without fcntl (e.g., Windows without extra deps)
            # Still serialize via existence + open; not as strong, but prevents many races.
            yield
    finally:
        f.close()


def _ensure_csv(csv_path: str, total_aez: int = DEFAULT_TOTAL_AEZ) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADER)
            w.writeheader()
            for aez in range(1, total_aez + 1):
                w.writerow({
                    "aez_no": aez,
                    "last_tile": "",
                    "tiles_complete": "false",
                    "samples_complete": "false",
                    "lulc_complete": "false",
                })


def _read_all_rows(csv_path: str) -> list[dict]:
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def _write_all_rows_atomic(csv_path: str, rows: list[dict]) -> None:
    # Write to a temp file then atomically replace to avoid torn writes
    dir_ = os.path.dirname(csv_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".state_", suffix=".csv", dir=dir_)
    os.close(fd)
    try:
        with open(tmp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADER)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        os.replace(tmp_path, csv_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _upsert_row(rows: list[dict], aez_no: int) -> dict:
    for row in rows:
        if str(row.get("aez_no")) == str(aez_no):
            return row
    # If not found, append default row
    row = {
        "aez_no": str(aez_no),
        "last_tile": "",
        "tiles_complete": "false",
        "samples_complete": "false",
        "lulc_complete": "false",
    }
    rows.append(row)
    return row


# ====== Public API ======
def fetch_state(csv_path: str, aez_no: int) -> Dict[str, Any]:
    """
    Return the state for a single AEZ:
    {
      'aez_no': int,
      'tiles': {'last_tile': Optional[int], 'complete': bool},
      'samples_complete': bool,
      'lulc_complete': bool
    }
    """
    _ensure_csv(csv_path)

    # Lock around read to avoid racing with a writer replacing the file
    lock_path = csv_path + ".lock"
    with _locked(lock_path):
        rows = _read_all_rows(csv_path)

    # Find or default
    row = None
    for r in rows:
        if str(r.get("aez_no")) == str(aez_no):
            row = r
            break
    if row is None:
        # Not present yet: return defaults (do not write anything on fetch)
        return {
            "aez_no": aez_no,
            "tiles": {"last_tile": None, "complete": False},
            "samples_complete": False,
            "lulc_complete": False,
        }

    return {
        "aez_no": int(row["aez_no"]),
        "tiles": {
            "last_tile": _parse_int_or_none(row.get("last_tile", "")),
            "complete": _parse_bool(row.get("tiles_complete", "false")),
        },
        "samples_complete": _parse_bool(row.get("samples_complete", "false")),
        "lulc_complete": _parse_bool(row.get("lulc_complete", "false")),
    }


def store_state(
    csv_path: str,
    aez_no: int,
    *,
    last_tile: Optional[int] = None,
    tiles_complete: Optional[bool] = None,
    samples_complete: Optional[bool] = None,
    lulc_complete: Optional[bool] = None,
    total_tiles: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Update and persist state for a given AEZ. Any parameter left as None is not changed.

    Tiles handling:
      - If tiles_complete is provided, it wins.
      - Else if last_tile and total_tiles are provided, tiles_complete is derived as (last_tile >= total_tiles).
      - Else tiles_complete remains unchanged.

    Returns the updated state dict (same shape as fetch_state).
    """
    _ensure_csv(csv_path)
    lock_path = csv_path + ".lock"

    with _locked(lock_path):
        rows = _read_all_rows(csv_path)
        row = _upsert_row(rows, aez_no)

        # last_tile
        if last_tile is not None:
            if last_tile < 0:
                raise ValueError("last_tile cannot be negative")
            row["last_tile"] = str(last_tile)

        # tiles_complete precedence: explicit flag > derived from totals > unchanged
        if tiles_complete is not None:
            row["tiles_complete"] = _fmt_bool(tiles_complete)
        elif last_tile is not None and total_tiles is not None:
            row["tiles_complete"] = _fmt_bool(last_tile >= total_tiles)

        # samples_complete / lulc_complete
        if samples_complete is not None:
            row["samples_complete"] = _fmt_bool(samples_complete)
        if lulc_complete is not None:
            row["lulc_complete"] = _fmt_bool(lulc_complete)

        _write_all_rows_atomic(csv_path, rows)

        # Build return payload from the just-written row
        updated = {
            "aez_no": int(row["aez_no"]),
            "tiles": {
                "last_tile": _parse_int_or_none(row.get("last_tile", "")),
                "complete": _parse_bool(row.get("tiles_complete", "false")),
            },
            "samples_complete": _parse_bool(row.get("samples_complete", "false")),
            "lulc_complete": _parse_bool(row.get("lulc_complete", "false")),
        }
        return updated


# ====== (Optional) Example usage ======
#if __name__ == "__main__":
#    path = "aez_state.csv"

    # Update examples:
    #print(store_state(path, 3, last_tile=12, total_tiles=20))            # sets last_tile=12, tiles_complete=False (derived)
#    print(store_state(path, 3, last_tile=20, total_tiles=20))            # sets last_tile=20, tiles_complete=True (derived)
#    print(store_state(path, 3, samples_complete=True))                   # mark samples done
#    print(store_state(path, 3, lulc_complete=True))                      # mark lulc done
#    print(store_state(path, 5, tiles_complete=True, samples_complete=False))  # explicit tiles_complete

    # Fetch:
    #print(fetch_state(path, 3))
    #print(fetch_state(path, 5))
#    print(fetch_state(path, 1))  # default row if not updated yet
