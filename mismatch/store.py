"""On-disk format for a view's logprobs: one .npz per view per model.

  logp     float32 [total_tokens]   log pi(x_t | prefix) for every completion token
  offsets  int64   [n_rows + 1]     row r spans logp[offsets[r]:offsets[r+1]]
  ids      int64   [n_rows]         corpus row id, so views can be aligned
  extra_*  optional per-token arrays (reference view stores entropy, top1 flag)
  meta     json string              exact settings that produced it

Rows are stored in corpus order; a view scored with --limit stores a prefix.
"""
import json
import numpy as np


def save_view(path, logp_rows, ids, meta, extra=None):
    lengths = np.array([len(r) for r in logp_rows], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    logp = np.concatenate([np.asarray(r, dtype=np.float32) for r in logp_rows]) if logp_rows else np.zeros(0, np.float32)
    arrays = {"logp": logp, "offsets": offsets, "ids": np.asarray(ids, dtype=np.int64),
              "meta": np.array(json.dumps(meta))}
    for k, v in (extra or {}).items():
        arrays[f"extra_{k}"] = np.concatenate([np.asarray(r, dtype=np.float32) for r in v])
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    import os
    os.replace(tmp, path)


class View:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.logp = z["logp"]
        self.offsets = z["offsets"]
        self.ids = z["ids"]
        self.meta = json.loads(str(z["meta"]))
        self.extra = {k[len("extra_"):]: z[k] for k in z.files if k.startswith("extra_")}
        self.n_rows = len(self.ids)

    def row(self, r):
        return self.logp[self.offsets[r]:self.offsets[r + 1]]

    def lengths(self):
        return np.diff(self.offsets)


def align(a, b):
    """Common rows (by id) of two views, in a's order. Returns (ids, a_slices, b_slices)."""
    pos_b = {int(i): r for r, i in enumerate(b.ids)}
    out_ids, ra, rb = [], [], []
    for r, i in enumerate(a.ids):
        if int(i) in pos_b:
            out_ids.append(int(i)); ra.append(r); rb.append(pos_b[int(i)])
    return out_ids, ra, rb


def flatten_pair(a, b):
    """Aligned flat arrays (logp_a, logp_b, row_index, position) over common rows."""
    ids, ra, rb = align(a, b)
    la, lb, rows, pos = [], [], [], []
    for k, (i, j) in enumerate(zip(ra, rb)):
        x, y = a.row(i), b.row(j)
        assert len(x) == len(y), f"row {ids[k]}: {len(x)} vs {len(y)} tokens"
        la.append(x); lb.append(y)
        rows.append(np.full(len(x), k, dtype=np.int64))
        pos.append(np.arange(len(x), dtype=np.int64))
    cat = lambda z: np.concatenate(z) if z else np.zeros(0)
    return cat(la), cat(lb), cat(rows), cat(pos), len(ids)
