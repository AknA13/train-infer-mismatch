"""store.py round trip and alignment across views with different row subsets."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mismatch.store import View, align, flatten_pair, save_view

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    rng = np.random.default_rng(1)
    rows = [rng.normal(size=n).astype(np.float32) for n in (3, 7, 1, 12)]
    with tempfile.TemporaryDirectory() as d:
        pa, pb = Path(d) / "a.npz", Path(d) / "b.npz"
        save_view(pa, rows, [0, 1, 2, 3], {"family": "hf"}, extra={"entropy": [np.abs(r) for r in rows]})
        save_view(pb, [rows[3] + 1, rows[1] + 1], [3, 1], {"family": "vllm"})   # subset, other order
        a, b = View(pa), View(pb)
        check("round trip logp", all(np.array_equal(a.row(i), rows[i]) for i in range(4)))
        check("meta json", a.meta == {"family": "hf"} and b.meta["family"] == "vllm")
        check("extra stored", "entropy" in a.extra and len(a.extra["entropy"]) == 23)
        ids, ra, rb = align(a, b)
        check("align common ids in a order", ids == [1, 3] and ra == [1, 3] and rb == [1, 0])
        la, lb, rws, pos, n = flatten_pair(a, b)
        check("flatten pair sizes", len(la) == 19 and n == 2 and pos.max() == 11)
        check("flatten pair values", np.allclose(lb - la, 1.0))
        check("row index", rws.tolist() == [0] * 7 + [1] * 12)
        # mismatched lengths must be a loud error, not a silent misalignment
        pc = Path(d) / "c.npz"; save_view(pc, [rows[1][:-1]], [1], {})
        try:
            flatten_pair(a, View(pc)); check("length mismatch raises", False)
        except AssertionError:
            check("length mismatch raises", True)
    print(); print("FAILED" if FAIL else "store OK"); return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
