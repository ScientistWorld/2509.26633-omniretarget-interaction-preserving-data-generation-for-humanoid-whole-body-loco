"""One-time extraction of the held-out test demonstrations into eval/test/data/.

Held out (never fetched by scripts/download.sh):
  * OMOMO large-box clips of subjects sub7 and sub16 (InterMimic-processed
    release, zip id 141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j on Google Drive),
  * climbing MoCap sequences 0, 2 and 3 of the paper's released demo data
    (holosoma_retargeting/demo_data/climb, commit c9cd097).
Stores the cropped windows (joints, object poses) and the terrain boxes as a
compressed npz; eval/test/hidden_tasks.py builds the instances from it.

Usage: python eval/test/build_test_data.py --omomo-zip OMOMO_new.zip --climb-dir .../demo_data/climb
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST_SUBJECTS = ("sub7", "sub16")
TEST_SEQUENCES = (0, 2, 3)
WINDOW, MIN_FRAMES = 150, 90


def load_pt_array(raw: bytes) -> np.ndarray:
    zf = zipfile.ZipFile(io.BytesIO(raw))
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]
    store = {}

    def rebuild(storage, offset, size, stride, *a):
        return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                               strides=tuple(s * storage.itemsize for s in stride)).copy()

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild
            if module == "torch" and name == "FloatStorage":
                return name
            if (module, name) == ("collections", "OrderedDict"):
                import collections
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"{module}.{name}")

        def persistent_load(self, pid):
            key = pid[2]
            if key not in store:
                store[key] = np.frombuffer(zf.read(f"{prefix}data/{key}"), dtype=np.float32)
            return store[key]

    return np.asarray(U(io.BytesIO(zf.read(pkl))).load())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omomo-zip", type=Path, required=True)
    ap.add_argument("--climb-dir", type=Path, required=True)
    args = ap.parse_args()
    z = zipfile.ZipFile(args.omomo_zip)
    arrays, meta = {}, {"omomo": [], "climb": []}
    for name in sorted(z.namelist()):
        base = Path(name).stem
        if "_largebox_" not in base or base.split("_")[0] not in TEST_SUBJECTS:
            continue
        a = load_pt_array(z.read(name))
        joints = a[:, 162:318].reshape(-1, 52, 3)
        p = a[:, 318:325]
        poses = np.concatenate([p[:, :3], p[:, [6]], p[:, 3:6]], axis=1)
        v = np.linalg.norm(np.diff(poses, axis=0), axis=1)
        tm = int(np.argmax(v > 0.0025)) if np.any(v > 0.0025) else 0
        s = max(0, tm - 30)
        e = min(len(joints), s + WINDOW)
        if e - s < MIN_FRAMES:
            continue
        arrays[f"omomo/{base}/joints"] = joints[s:e].astype(np.float32)
        arrays[f"omomo/{base}/poses"] = poses[s:e].astype(np.float32)
        meta["omomo"].append(base)
    for sid in TEST_SEQUENCES:
        d = args.climb_dir / f"mocap_climb_seq_{sid}"
        J = np.load(sorted(glob.glob(str(d / "*_joint_positions_*.npy")))[0])[::4].astype(np.float32)
        arrays[f"climb/{sid}/joints"] = J
        boxes = []
        for f in sorted(glob.glob(str(d / "box_models" / "*.obj"))):
            V = np.array([ln.split()[1:4] for ln in open(f) if ln.startswith("v ")], dtype=float)
            boxes.append(V.tolist())
        meta["climb"].append({"sequence": sid, "box_vertices": boxes})
    out = HERE / "data" / "test_demonstrations.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(meta['omomo'])} OMOMO windows, "
          f"{len(meta['climb'])} climbing sequences")


if __name__ == "__main__":
    main()
