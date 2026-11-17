"""Shortcut submissions run through the hidden test (iteration 13, 3D gym).

Each writes outputs for every input it is given (unless the shortcut is to omit some).

constant_pose        valid standing robot at the demonstrator's starting position, never moving
impossible_q         every coordinate 1e200
nan                  NaN configuration
malformed            zero frames per instance
empty                writes nothing
partial              reference outputs for the box-carrying instances only (omits the others)
joint_violation      keypoint-IK outputs with one joint 0.1 rad beyond its limit on one frame
velocity_violation   reference outputs with one joint jumping by 1.5 rad for one frame
unnormalized_quat    reference outputs with the base quaternion scaled by 2
teleport             reference outputs with the base shifted by 1 m on one frame
score_forgery        keypoint-IK outputs + claimed perfect metrics in the output files and
                     prewritten perfect scoring/scores_*.json
always_contact       reference + contact costs that pull both hands onto the box / terrain on
                     every frame (tries to buy contact preservation without the demonstration)

Usage: python redteam/hardening3d/shortcuts.py <name> <input_dir> <output_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _inputs(d: Path):
    for p in sorted(d.glob("*.json")):
        yield p, json.loads(p.read_text())


def _write(out: Path, name: str, q, extra: dict | None = None):
    data = {"q": np.asarray(q).tolist()}
    data.update(extra or {})
    (out / name).write_text(json.dumps(data))


CACHE = ROOT / "redteam" / "hardening3d" / "results" / "official" / "outputs"


def _cached(method: str, inst):
    """Output of a shipped method from the official run (deterministic), if saved."""
    for split in ("test", "train"):
        f = CACHE / split / method / f"{inst['instance_id']}.json"
        if f.exists():
            return np.asarray(json.loads(f.read_text())["q"], dtype=float)
    return None


def _kik(inst):
    q = _cached("keypoint_ik", inst)
    if q is not None:
        return q
    from baseline.keypoint_ik import retarget
    return retarget(inst)


def _ref(inst, **kw):
    q = _cached("omniretarget", inst) if not kw else None
    if q is not None:
        return q
    from method.omniretarget import retarget
    return retarget(inst, **kw)


def constant_pose(inst):
    names = inst["joint_names"]
    J = np.asarray(inst["human_joints"], dtype=float)
    pel = J[0, names.index("Pelvis" if "Pelvis" in names else "Hips")] * float(inst["alpha"])
    hips = ("L_Hip", "R_Hip") if "L_Hip" in names else ("LeftUpLeg", "RightUpLeg")
    fwd = np.cross(J[0, names.index(hips[0])] - J[0, names.index(hips[1])], [0, 0, 1.0])
    yaw = np.arctan2(fwd[1], fwd[0])
    q = np.zeros(36)
    q[:3] = [pel[0], pel[1], 0.793]
    q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    return np.repeat(q[None], int(inst["num_frames"]), axis=0)


def always_contact(inst):
    import method.omniretarget as M

    class R(M.Retargeter):
        def _demo_contacts(self):
            c = super()._demo_contacts()
            return {k: (np.ones_like(v) if k.endswith("hand") else v) for k, v in c.items()}

    kw = {"objective": "mesh", "contact_weight": 100.0}
    if inst.get("augmentation") and inst["task"] == "object":
        q_nom = R(inst, inst["nominal_scene"], **kw).run()
        return R(inst, inst["target_scene"], **kw).run(q_nominal=q_nom)
    return R(inst, inst["target_scene"], **kw).run()


def main() -> None:
    name, inp, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if name == "empty":
        return
    if name == "score_forgery":
        perfect = {"penetration_duration": 0.0, "max_penetration_cm": 0.0, "foot_skating_duration": 0.0,
                   "max_foot_skating_cm_s": 0.0, "contact_preservation": 1.0, "spurious_contact_duration": 0.0,
                   "pelvis_path_error_cm": 0.0, "pose_error_cm": 0.0}
        fake = {"experiments": {e: {"results": {"sc_score_forgery": perfect}} for e in
                                ("object_interaction_kinematics", "terrain_interaction_kinematics",
                                 "augmented_interaction_kinematics")}}
        for f in ("scores_test.json", "scores_train.json"):
            (ROOT / "scoring" / f).write_text(json.dumps(fake))
    for p, inst in _inputs(inp):
        T = int(inst["num_frames"])
        if name == "constant_pose":
            q = constant_pose(inst)
        elif name == "impossible_q":
            q = np.full((T, 36), 1e200)
        elif name == "nan":
            q = np.full((T, 36), np.nan)
        elif name == "malformed":
            q = np.zeros((0, 36))
        elif name == "partial":
            if inst["task"] != "object" or inst.get("augmentation"):
                continue
            q = _ref(inst)
        elif name == "joint_violation":
            q = _kik(inst)
            q[T // 2, 7 + 3] = 2.8798 + 0.1     # left knee beyond its upper limit
        elif name == "velocity_violation":
            q = _ref(inst)
            q[T // 2, 7 + 12] += 1.5 if q[T // 2, 7 + 12] < 0 else -1.5   # waist yaw jump (within limits)
        elif name == "unnormalized_quat":
            q = _ref(inst)
            q[:, 3:7] *= 2.0
        elif name == "teleport":
            q = _ref(inst)
            q[T // 2, 0] += 1.0
        elif name == "score_forgery":
            q = _kik(inst)
            _write(out, p.name, q, {"metrics": {"contact_preservation": 1.0, "penetration_duration": 0.0},
                                    "valid": True, "cpu_s": 0.0})
            continue
        elif name == "always_contact":
            q = always_contact(inst)
        else:
            raise SystemExit(f"unknown shortcut {name}")
        _write(out, p.name, q)


if __name__ == "__main__":
    main()
