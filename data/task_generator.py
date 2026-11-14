"""Procedural human demonstrations for the interaction-retargeting benchmark.

Each instance is a short (2.6-3.2 s, 30 Hz) demonstration by a planar human
whose overall size (0.90-1.15 times the robot) and individual segment
proportions differ from the robot's.  The human side-steps along +x with a
planted-foot gait (stance feet are exactly stationary; swing feet lift and
land).

Task types
----------
object_carry   The human carries a box overhead with both hands on the box
               sides while stepping; the box also moves up and down.
terrain_step   The human steps up a ramp onto a platform and braces the right
               hand on top of a tall column standing on the platform.

Scenes
------
``source_scene``   what the human interacted with (human scale).
``nominal_scene``  the same scene brought to the robot's scale by direct
                   scaling (paper App. A, Eq. 7/8): every terrain vertex and
                   the object's trajectory are scaled by
                   alpha = h_robot / h_human about the ground point below the
                   demonstrator's initial pelvis.  The object keeps its true
                   size (the robot handles the same object), so scaled human
                   hand positions do not land exactly on it.
``target_scene``   where the robot must act.  For nominal instances it equals
                   ``nominal_scene``; for augmented instances it is a modified
                   copy (paper Sec. III-B / App. C): the box is resized and
                   displaced with an exponentially decaying offset (Eq. 14),
                   or the platform (and the column on it) is raised or lowered.

Heights h are foot-to-neck heights at full extension (torso + thigh + shin).

Outputs per instance
--------------------
``input``       what a retargeting method receives (source keypoints, the
                three scenes, timing).
``annotation``  evaluation labels: desired foot-stance frames, desired hand
                contacts, the scaling (alpha, anchor) and generator metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.planar_robot import KEYPOINTS, PlanarHumanoid
from data.scene_geometry import box_sdf, terrain_height, terrain_sdf

DT = 1.0 / 30.0
ROBOT = PlanarHumanoid()
EXPERIMENT_OF = {
    ("object_carry", False): "object_interaction_kinematics",
    ("terrain_step", False): "terrain_interaction_kinematics",
    ("object_carry", True): "augmented_interaction_kinematics",
    ("terrain_step", True): "augmented_interaction_kinematics",
}
HUMAN_LINKS = [(0, 1), (1, 6), (1, 9), (0, 2), (2, 3), (0, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)]

# Augmentation ranges (paper Sec. III-B, App. C).
AUG_LEVELS = {
    "small": {"k": (0.85, 1.20), "dpx": (0.03, 0.08), "dpz": 0.05, "tau": (1.0, 3.0),
              "H_mult": (0.5, 1.6), "H_clip": (0.05, 0.32), "dH_min": 0.04},
    "large": {"k": (0.75, 1.35), "dpx": (0.06, 0.16), "dpz": 0.08, "tau": (1.0, 3.0),
              "H_mult": (0.4, 1.8), "H_clip": (0.04, 0.34), "dH_min": 0.05},
}
# scene_mode "robot": nominal/target scenes are the demonstration scene at the
# robot's scale (above).  "human": they keep the demonstrator's scale.
DEFAULT_CONFIG = {"scene_mode": "robot", "prop_jitter": 0.06, "aug_level": "large", "object_terrain": False}


class Reject(Exception):
    pass


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def _ease(u):
    u = np.clip(u, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * u))


def _two_link(root: np.ndarray, target: np.ndarray, a: float, b: float, outward: float) -> tuple[np.ndarray, float]:
    """Middle joint for a 2-link chain; ``outward`` = -1 picks smaller x, +1 larger x.

    Returns (joint, reach_ratio) with reach_ratio = |target-root| / (a+b).
    """
    d = target - root
    L = float(np.linalg.norm(d))
    ratio = L / (a + b)
    Lc = float(np.clip(L, abs(a - b) + 1e-6, (a + b) * 0.9999))
    u = d / max(L, 1e-9)
    along = (a * a - b * b + Lc * Lc) / (2 * Lc)
    h = np.sqrt(max(a * a - along * along, 0.0))
    n = np.array([-u[1], u[0]])
    m1 = root + along * u + h * n
    m2 = root + along * u - h * n
    if outward < 0:
        return (m1 if m1[0] <= m2[0] else m2), ratio
    return (m1 if m1[0] >= m2[0] else m2), ratio


def _human_dims(rng: np.random.Generator, s: float, jitter: float) -> dict:
    j = lambda: float(rng.uniform(1.0 - jitter, 1.0 + jitter))  # noqa: E731
    return {
        "thigh": ROBOT.thigh * s * j(),
        "shin": ROBOT.shin * s * j(),
        "torso": ROBOT.torso * s * j(),
        "shoulder_x": 0.19 * s * j(),
        "shoulder_drop": 0.07 * s * j(),
        "upper_arm": ROBOT.upper_arm * s * j(),
        "forearm": ROBOT.forearm * s * j(),
    }


def _gait(rng, N, x0, w0, step_range, s):
    """Foot x/z bases and swing phases for a side-stepping gait."""
    d0 = int(rng.integers(4, 8))
    steps = []
    t = d0
    foot = "R"
    x_l, x_r = x0 - w0, x0 + w0
    while True:
        tsw = int(rng.integers(8, 12))
        if t + tsw + 3 > N:
            break
        if foot == "R":
            ell = float(rng.uniform(*step_range)) * s
            x_r += ell
        else:  # trailing foot restores a (jittered) stance width
            ell = (x_r - x_l) - 2 * w0 * float(rng.uniform(0.92, 1.08))
            if ell < 0.03:
                raise Reject("degenerate trailing step")
            x_l += ell
        steps.append((foot, t, tsw, ell))
        t += tsw + int(rng.integers(2, 5))
        foot = "L" if foot == "R" else "R"
    if len(steps) < 4:
        raise Reject("too few steps")
    return steps


def _foot_tracks(steps, N, x_start, terrain, lift_rng):
    """Per-foot x, z and planted flags given a step list."""
    xs = {"L": np.zeros(N), "R": np.zeros(N)}
    zs = {"L": np.zeros(N), "R": np.zeros(N)}
    base = {"L": np.zeros(N), "R": np.zeros(N)}
    planted = {"L": np.ones(N, dtype=bool), "R": np.ones(N, dtype=bool)}
    cur = dict(x_start)
    events = {"L": [], "R": []}
    for foot, t0, tsw, ell in steps:
        events[foot].append((t0, tsw, ell, float(lift_rng.uniform(0.04, 0.08))))
    for foot in ("L", "R"):
        x = cur[foot]
        evs = events[foot]
        k = 0
        t = 0
        while t < N:
            if k < len(evs) and t == evs[k][0]:
                t0, tsw, ell, lift = evs[k]
                xs_, xe_ = x, x + ell
                zs_, ze_ = float(terrain_height(xs_, terrain)), float(terrain_height(xe_, terrain))
                for i in range(1, tsw + 1):
                    if t0 + i >= N:
                        break
                    phi = i / tsw
                    xi = xs_ + (xe_ - xs_) * _ease(phi)
                    bi = zs_ + (ze_ - zs_) * _ease(phi)
                    zi = max(float(terrain_height(xi, terrain)), bi) + lift * np.sin(np.pi * phi)
                    xs[foot][t0 + i] = xi
                    zs[foot][t0 + i] = zi
                    base[foot][t0 + i] = max(bi, float(terrain_height(xi, terrain)))
                    planted[foot][t0 + i] = i == tsw
                xs[foot][t0] = x
                zs[foot][t0] = float(terrain_height(x, terrain))
                base[foot][t0] = zs[foot][t0]
                x = xe_
                t = t0 + tsw + 1
                k += 1
                continue
            xs[foot][t] = x
            zs[foot][t] = float(terrain_height(x, terrain))
            base[foot][t] = zs[foot][t]
            t += 1
    return xs, zs, base, planted


def _robot_can_hold(targets, pelvis_x, ground, box=None) -> bool:
    """Coarse check that some robot posture reaches all hand targets.

    ``targets`` maps 'L'/'R' to target points; ``box`` = (center, size) blocks the head.
    """
    reach = 0.95 * (ROBOT.upper_arm + ROBOT.forearm)
    for pz in np.linspace(0.45, 0.78, 12):
        for tor in np.linspace(-0.35, 0.35, 9):
            pel = np.array([pelvis_x, ground + pz])
            neck = pel + ROBOT.torso * np.array([np.sin(tor), np.cos(tor)])
            sh = {"L": neck + np.array([-ROBOT.shoulder_x, -ROBOT.shoulder_drop]),
                  "R": neck + np.array([ROBOT.shoulder_x, -ROBOT.shoulder_drop])}
            if box is not None:
                pts = np.stack([pel + u * (neck - pel) for u in np.linspace(0, 1, 7)] + [sh["L"], sh["R"]])
                if np.min(box_sdf(pts, box[0], box[1])) < 0.02:
                    continue
            if all(np.linalg.norm(targets[k] - sh[k]) <= reach for k in targets):
                return True
    return False


def _attempt(rng: np.random.Generator, kind: str, augmented: bool, cfg: dict) -> tuple[dict, dict]:
    s = float(rng.uniform(0.90, 1.15))
    dims = _human_dims(rng, s, float(cfg["prop_jitter"]))
    leg = dims["thigh"] + dims["shin"]
    arm = dims["upper_arm"] + dims["forearm"]
    N = int(rng.integers(78, 97))
    w0 = float(rng.uniform(0.14, 0.20)) * s
    x0 = float(rng.uniform(-0.7, -0.4))
    step_range = (0.10, 0.20) if kind == "object_carry" else (0.14, 0.24)
    steps = _gait(rng, N, x0, w0, step_range, s)
    x_start = {"L": x0 - w0, "R": x0 + w0}

    # Provisional flat tracks to locate the final right foot.
    flat = [[-4.0, 0.0], [4.0, 0.0]]
    xs_f, _, _, _ = _foot_tracks(steps, N, x_start, flat, np.random.default_rng(0))
    x_rf = float(xs_f["R"][-1])
    x_r0 = x_start["R"]

    ramp = kind == "terrain_step" or bool(cfg.get("object_terrain"))
    if ramp:
        W_r = float(rng.uniform(0.08, 0.16))
        lo, hi = x_r0 + 0.04, x_rf - W_r - 0.04
        if hi <= lo:
            raise Reject("no room for ramp")
        x_r = float(rng.uniform(lo, hi))
        H = float(rng.uniform(0.10, 0.24))
    if kind == "terrain_step":
        x_c = x_rf + float(rng.uniform(0.12, 0.18))
        W_c = float(rng.uniform(0.14, 0.22))
        terrain_src = None  # column height filled in after the body is built
        tmp_terrain = [[-4.0, 0.0], [x_r, 0.0], [x_r + W_r, H], [x_c, H], [x_c + 0.02, H + 2.0],
                       [x_c + W_c - 0.02, H + 2.0], [x_c + W_c, H], [4.0, H]]
    elif ramp:  # box carried up a ramp onto a platform (robot-object-terrain interaction)
        terrain_src = [[-4.0, 0.0], [x_r, 0.0], [x_r + W_r, H], [4.0, H]]
        tmp_terrain = terrain_src
    else:
        terrain_src = flat
        tmp_terrain = flat
    lift_seed = int(rng.integers(1 << 30))
    xs, zs, base, planted = _foot_tracks(steps, N, x_start, tmp_terrain, np.random.default_rng(lift_seed))

    # Pelvis trajectory.
    px = _smooth(0.5 * (xs["L"] + xs["R"]), 9)
    g = _smooth(0.5 * (base["L"] + base["R"]), 9)
    c = float(rng.uniform(0.93, 0.97))
    h_nom = np.sqrt((c * leg) ** 2 - w0 ** 2)
    period = float(rng.uniform(0.5, 0.9)) * N
    bob = float(rng.uniform(0.0, 0.02)) * 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / period))
    pz = g + h_nom - bob
    cap = np.full(N, np.inf)
    for f in ("L", "R"):
        dx = px - xs[f]
        if np.any(np.abs(dx) >= 0.985 * leg):
            raise Reject("feet too far")
        cap = np.minimum(cap, zs[f] + np.sqrt((0.985 * leg) ** 2 - dx ** 2))
    pz = np.minimum(_smooth(np.minimum(pz, cap), 5), cap)

    # Torso and right-hand reach timing.
    t_idx = np.arange(N)
    sway = float(rng.uniform(0.0, 0.05)) * np.sin(2 * np.pi * t_idx / float(rng.uniform(0.6, 1.0) * N) + float(rng.uniform(0, 2 * np.pi)))
    tor = sway.copy()
    if kind == "terrain_step":
        t_arrive = N - int(rng.integers(22, 35))
        t_reach = int(rng.integers(10, 17))
        prog = _ease((t_idx - (t_arrive - t_reach)) / t_reach)
        tor = tor + float(rng.uniform(0.04, 0.14)) * prog
    pel = np.stack([px, pz], axis=1)
    neck = pel + dims["torso"] * np.stack([np.sin(tor), np.cos(tor)], axis=1)
    lsh = neck + np.array([-dims["shoulder_x"], -dims["shoulder_drop"]])
    rsh = neck + np.array([dims["shoulder_x"], -dims["shoulder_drop"]])

    lfoot = np.stack([xs["L"], zs["L"]], axis=1)
    rfoot = np.stack([xs["R"], zs["R"]], axis=1)
    lknee = np.zeros((N, 2))
    rknee = np.zeros((N, 2))
    for t in range(N):
        lknee[t], r1 = _two_link(pel[t], lfoot[t], dims["thigh"], dims["shin"], -1)
        rknee[t], r2 = _two_link(pel[t], rfoot[t], dims["thigh"], dims["shin"], +1)
        if max(r1, r2) > 0.999:
            raise Reject("leg overstretched")

    hand_contact = {"left_hand": [None] * N, "right_hand": [None] * N}
    lhand = np.zeros((N, 2))
    rhand = np.zeros((N, 2))
    obj_src = None
    if kind == "object_carry":
        w = float(rng.uniform(0.34, 0.54))
        hgt = float(rng.uniform(0.14, 0.26))
        gap = float(rng.uniform(0.05, 0.10))
        press_amp = float(rng.uniform(0.0, 0.08))
        press_f = float(rng.uniform(0.3, 0.8))
        press = press_amp * 0.5 * (1 - np.cos(2 * np.pi * press_f * t_idx / N))
        off_x = float(rng.uniform(-0.02, 0.02))
        cx = neck[:, 0] + off_x
        cz = neck[:, 1] + gap + hgt / 2 + press
        centers = np.stack([cx, cz], axis=1)
        dl, dr = rng.uniform(-0.25, 0.25, size=2) * hgt
        lhand = centers + np.array([-w / 2, dl])
        rhand = centers + np.array([w / 2, dr])
        obj_src = {"size": [w, hgt], "center": centers}
        hand_contact = {"left_hand": ["object"] * N, "right_hand": ["object"] * N}
    else:
        H_c = float(rsh[-1, 1] - rng.uniform(0.20, 0.32) * s)
        if H_c < H + 0.45:
            raise Reject("column too low")
        terrain_src = [[-4.0, 0.0], [x_r, 0.0], [x_r + W_r, H], [x_c, H], [x_c + 0.02, H_c],
                       [x_c + W_c - 0.02, H_c], [x_c + W_c, H], [4.0, H]]
        contact_pt = np.array([x_c + W_c * float(rng.uniform(0.3, 0.6)), H_c])
        swing_amp = float(rng.uniform(0.0, 0.10))
        sw = swing_amp * np.sin(2 * np.pi * t_idx / float(rng.uniform(0.5, 0.9) * N))
        hang_l = lsh + 0.93 * arm * np.stack([np.sin(-0.10 + sw), -np.cos(-0.10 + sw)], axis=1)
        hang_r = rsh + 0.93 * arm * np.stack([np.sin(0.10 - sw), -np.cos(0.10 - sw)], axis=1)
        lhand = hang_l
        # Reach up first, then over onto the column top (avoids the column edge).
        via = np.array([x_c - 0.05, H_c + 0.07])
        e_up = _ease(prog / 0.6)[:, None]
        e_over = _ease((prog - 0.6) / 0.4)[:, None]
        rhand = hang_r + e_up * (via - hang_r) + e_over * (contact_pt - via)
        for t in range(N):
            if t >= t_arrive:
                rhand[t] = contact_pt
                hand_contact["right_hand"][t] = "terrain"

    lelb = np.zeros((N, 2))
    relb = np.zeros((N, 2))
    for t in range(N):
        lelb[t], r1 = _two_link(lsh[t], lhand[t], dims["upper_arm"], dims["forearm"], -1)
        relb[t], r2 = _two_link(rsh[t], rhand[t], dims["upper_arm"], dims["forearm"], +1)
        if max(r1, r2) > 0.97:
            raise Reject("arm overstretched")

    kp = np.stack([pel, neck, lknee, lfoot, rknee, rfoot, lsh, lelb, lhand, rsh, relb, rhand], axis=1)

    # The demonstration itself must be collision-free in its own scene.
    u = np.linspace(0, 1, 15)
    a = kp[:, [i for i, _ in HUMAN_LINKS]]
    b = kp[:, [j for _, j in HUMAN_LINKS]]
    samples = (a[:, :, None] + (b - a)[:, :, None] * u[:, None]).reshape(N, -1, 2)
    if np.min(terrain_sdf(samples, terrain_src)) < -0.003:
        raise Reject("human penetrates terrain")
    if obj_src is not None:
        for t in range(N):
            if np.min(box_sdf(samples[t], obj_src["center"][t], obj_src["size"])) < -0.003:
                raise Reject("human penetrates box")

    # Nominal scene at the robot's scale (direct scaling, App. A Eq. 7/8).
    h_human = dims["torso"] + dims["thigh"] + dims["shin"]
    alpha = ROBOT.height / h_human if cfg["scene_mode"] == "robot" else 1.0
    anchor = np.array([float(px[0]), float(terrain_height(px[0], terrain_src))])

    def to_robot(p):
        return anchor + alpha * (np.asarray(p, dtype=float) - anchor)

    terrain_nom = to_robot(np.asarray(terrain_src, dtype=float))
    obj_nom = None if obj_src is None else {"size": np.asarray(obj_src["size"], dtype=float),
                                             "center": to_robot(obj_src["center"])}

    # Target scene (augmentation of the nominal scene, Sec. III-B / App. C).
    A = AUG_LEVELS[cfg["aug_level"]]
    terrain_tgt = terrain_nom.copy()
    obj_tgt = None if obj_nom is None else {"size": obj_nom["size"].copy(), "center": obj_nom["center"].copy()}
    aug_info = {}
    if augmented and kind == "object_carry":
        k = float(rng.uniform(*A["k"]))
        dpx = float(rng.choice([-1, 1]) * rng.uniform(*A["dpx"]))
        dpz = float(rng.uniform(-A["dpz"], A["dpz"]))
        tau = float(rng.uniform(*A["tau"]))
        decay = np.exp(-t_idx * DT / tau)
        obj_tgt = {"size": obj_nom["size"] * k,
                   "center": obj_nom["center"] + np.stack([dpx * decay, dpz * decay], axis=1)}
        aug_info = {"size_scale": k, "offset": [dpx, dpz], "tau": tau}
    elif augmented and kind == "terrain_step":
        H_nom = float(terrain_nom[3, 1] - terrain_nom[1, 1])  # platform height in the nominal scene
        H2 = float(np.clip(H_nom * rng.uniform(*A["H_mult"]), *A["H_clip"]))
        if abs(H2 - H_nom) < A["dH_min"]:
            raise Reject("augmentation too small")
        dH = H2 - H_nom
        x_ramp = float(terrain_nom[1, 0])
        terrain_tgt = np.array([[x, z + dH if (x > x_ramp + 1e-9) else z] for x, z in terrain_nom])
        aug_info = {"platform_height": [H_nom, H2]}

    # Desired robot contacts must be physically attainable in the target scene.
    for t in range(N):
        pel_x = float(to_robot(pel[t])[0])
        if kind == "object_carry":
            cen, size = obj_tgt["center"][t], obj_tgt["size"]
            sc = size / np.asarray(obj_src["size"], dtype=float)
            targets = {"L": cen + (lhand[t] - obj_src["center"][t]) * sc,
                       "R": cen + (rhand[t] - obj_src["center"][t]) * sc}
            if not _robot_can_hold(targets, pel_x, float(terrain_height(pel_x, terrain_tgt)), (cen, size)):
                raise Reject("robot cannot hold the box")
        elif hand_contact["right_hand"][t] == "terrain":
            x_h = float(to_robot(rhand[t])[0])
            targets = {"R": np.array([x_h, float(terrain_height(x_h, terrain_tgt))])}
            if not _robot_can_hold(targets, pel_x, float(terrain_height(pel_x, terrain_tgt))):
                raise Reject("robot cannot reach the column")

    stance = {"left_foot": planted["L"].tolist(), "right_foot": planted["R"].tolist()}

    def scene(terrain, obj):
        return {"terrain": np.round(np.asarray(terrain, dtype=float), 6).tolist(),
                "object": None if obj is None else {
                    "size": [round(float(v), 6) for v in obj["size"]],
                    "center": np.round(np.asarray(obj["center"]), 6).tolist()}}

    inp = {
        "task_type": kind,
        "dt": DT,
        "num_frames": N,
        "keypoint_names": KEYPOINTS,
        "source_keypoints": np.round(kp, 6).tolist(),
        "source_scene": scene(terrain_src, obj_src),
        "nominal_scene": scene(terrain_nom, obj_nom),
        "target_scene": scene(terrain_tgt, obj_tgt),
    }
    ann = {
        "experiment": EXPERIMENT_OF[(kind, augmented)],
        "stance": stance,
        "hand_contact": hand_contact,
        "alpha": float(alpha),
        "anchor": [float(v) for v in anchor],
        "human_scale": s,
        "augmented": bool(augmented),
        "augmentation": aug_info,
    }
    return inp, ann


def make_instance(master_seed: int, index: int, kind: str, augmented: bool, instance_id: str,
                  cfg: dict | None = None) -> tuple[dict, dict]:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    rng = np.random.default_rng([master_seed, index])
    for _ in range(500):
        try:
            inp, ann = _attempt(rng, kind, augmented, cfg)
        except Reject:
            continue
        inp = {"instance_id": instance_id, **inp}
        ann = {"instance_id": instance_id, **ann}
        return inp, ann
    raise RuntimeError(f"could not generate instance {instance_id}")


def make_split(master_seed: int, per_experiment: int, prefix: str, cfg: dict | None = None) -> list[tuple[dict, dict]]:
    out = []
    idx = 0
    plan = []
    for i in range(per_experiment):
        plan.append(("object_carry", False, f"{prefix}_obj_{i:02d}"))
    for i in range(per_experiment):
        plan.append(("terrain_step", False, f"{prefix}_ter_{i:02d}"))
    for i in range(per_experiment):
        kind = "object_carry" if i % 2 == 0 else "terrain_step"
        plan.append((kind, True, f"{prefix}_aug_{i:02d}"))
    for kind, aug, iid in plan:
        out.append(make_instance(master_seed, idx, kind, aug, iid, cfg))
        idx += 1
    return out


TRAIN_MASTER_SEED = 20250930
TRAIN_PER_EXPERIMENT = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the public train instances.")
    parser.add_argument("--out", type=Path, default=Path("data/tasks/train"))
    parser.add_argument("--per-experiment", type=int, default=TRAIN_PER_EXPERIMENT)
    parser.add_argument("--seed", type=int, default=TRAIN_MASTER_SEED)
    args = parser.parse_args()
    (args.out / "inputs").mkdir(parents=True, exist_ok=True)
    (args.out / "annotations").mkdir(parents=True, exist_ok=True)
    for inp, ann in make_split(args.seed, args.per_experiment, "train"):
        iid = inp["instance_id"]
        (args.out / "inputs" / f"{iid}.json").write_text(json.dumps(inp))
        (args.out / "annotations" / f"{iid}.json").write_text(json.dumps(ann))
        print(iid, ann["experiment"], inp["num_frames"])


if __name__ == "__main__":
    main()
