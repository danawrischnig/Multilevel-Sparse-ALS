import os
import numpy as np
import tempfile
from pathlib import Path


def save_tensortrain(fn, key, components):
    packed = pack_tensortrain(components)
    fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if fn.exists():
        with np.load(str(fn)) as z:
            for k in z.files:
                data[k] = z[k]

    data[key] = packed

    tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(fn.parent), suffix=".npz")
    tmp_path = Path(tmp.name)
    try:
        tmp.close()
        np.savez_compressed(tmp_path, **data)
        os.replace(tmp_path, fn)
    finally:
        if tmp_path.exists() and fn != tmp_path:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def pack_tensortrain(components):
    # components[i]: (r_i, d, r_i)
    d = components[0].shape[1]
    assert all(C.shape[1] == d for C in components), "d muss konstant sein."
    ranks = [C.shape[0] for C in components] + [1]  # letzter Rang ist 1
    rmax = max(ranks)
    n = len(components)

    packed = np.zeros((n, rmax, d, rmax), dtype=components[0].dtype)
    for i, C in enumerate(components):
        r_left, r_right = ranks[i : i + 2]
        packed[i, :r_left, :, :r_right] = C
    return packed


def unpack_tensortrain(packed):
    # packed: (n, rmax, d, rmax), ranks: (n,)
    ranks = reconstruct_ranks(packed)
    n = packed.shape[0]
    comps = []
    for i in range(n):
        r_left, r_right = ranks[i : i + 2]
        comps.append(packed[i, :r_left, :, :r_right].copy())
    return comps


def reconstruct_ranks(packed: np.ndarray, tol=0.0) -> np.ndarray:
    """
    packed: shape (n, rmax, d, rmax)
    returns: ranks of length n+1
    """
    n, rmax1, d, rmax2 = packed.shape
    if rmax1 != rmax2:
        raise ValueError("packed must have shape (n, rmax, d, rmax)")

    left = np.zeros(n, dtype=int)  # ranks[i]
    right = np.zeros(n, dtype=int)  # ranks[i+1]

    for i in range(n):
        C = packed[i]  # shape (rmax, d, rmax)
        nz = np.abs(C) > tol

        # last nonzero along the first-rank axis (rows of C)
        row_has = nz.any(axis=(1, 2))  # (rmax,)
        idx = np.flatnonzero(row_has)
        left[i] = (idx[-1] + 1) if idx.size else 0

        # last nonzero along the last-rank axis (cols of C)
        col_has = nz.any(axis=(0, 1))  # (rmax,)
        idx = np.flatnonzero(col_has)
        right[i] = (idx[-1] + 1) if idx.size else 0

    # ranks[i] should equal left[i] and ranks[i+1] should equal right[i].
    # To be robust to tiny numerical noise, reconcile middles with max().
    ranks = np.empty(n + 1, dtype=int)
    ranks[0] = left[0]
    for i in range(1, n):
        ranks[i] = max(left[i], right[i - 1])
    ranks[n] = right[-1]

    return ranks


def load_tensortrain(fn, key):
    """
    Lädt ein gepacktes Tensor-Train aus einer .npz-Datei und gibt die Komponentenliste zurück.
    """
    fn = Path(fn)
    if not fn.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {fn}")

    with np.load(str(fn)) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' nicht in {fn}. Verfügbar: {list(z.files)}")
        packed = z[key].copy()  # in RAM kopieren, bevor der Kontext schließt

    return unpack_tensortrain(packed)
