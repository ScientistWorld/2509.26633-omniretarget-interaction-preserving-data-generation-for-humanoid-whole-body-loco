"""Torch-free reader for the InterMimic-processed OMOMO ``.pt`` files.

Each file is a ``torch.save``d float tensor of shape (T, 591) (30 fps).  The
layout used here follows the paper authors' public loader
(holosoma_retargeting ``load_intermimic_data``):
  columns 162:318 -> 52 SMPL-H joint positions (world frame, metres, z up)
  columns 318:325 -> object pose [x, y, z, qx, qy, qz, qw]
We parse the zip/pickle container directly so that torch is not required.
"""

from __future__ import annotations

import io
import pickle
import zipfile

import numpy as np

SMPLH_JOINTS = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3", "L_Pinky1", "L_Pinky2",
    "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Thorax",
    "R_Shoulder", "R_Elbow", "R_Wrist", "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2",
    "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1",
    "R_Thumb2", "R_Thumb3",
]

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32}


class _Storage:
    def __init__(self, dtype, key):
        self.dtype, self.key = dtype, key


def _rebuild_tensor_v2(storage, offset, size, stride, *args):
    arr = storage
    n = int(np.prod(size)) if len(size) else 1
    flat = arr[offset: offset + max(n, 1) if not stride else None]
    if len(size) == 0:
        return flat[0]
    return np.lib.stride_tricks.as_strided(
        arr[offset:], shape=tuple(size), strides=tuple(s * arr.itemsize for s in stride)).copy()


def load_pt_array(raw: bytes) -> np.ndarray:
    """Return the tensor stored in a torch-saved zip (bytes) as a numpy array."""
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    pkl = [n for n in names if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]
    storages: dict[str, np.ndarray] = {}

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch" and name in _DTYPES:
                return name
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _loc, _numel = pid
            if key not in storages:
                storages[key] = np.frombuffer(zf.read(f"{prefix}data/{key}"), dtype=_DTYPES[stype])
            return storages[key]

    return np.asarray(_U(io.BytesIO(zf.read(pkl))).load())


def parse_intermimic(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 591) -> joints (T, 52, 3), object pose (T, 7) as [x, y, z, qw, qx, qy, qz]."""
    joints = arr[:, 162:162 + 52 * 3].reshape(-1, 52, 3).astype(np.float64)
    pose = arr[:, 318:325].astype(np.float64)  # x y z qx qy qz qw
    obj = np.concatenate([pose[:, :3], pose[:, [6]], pose[:, 3:6]], axis=1)
    return joints, obj
