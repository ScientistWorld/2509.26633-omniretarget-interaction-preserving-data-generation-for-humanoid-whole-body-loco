"""Hidden test instances for the interaction-retargeting benchmark.

Self-contained copy of the public demonstration generator; the test split is
drawn with a private master seed and is only materialised at test time.
"""

from __future__ import annotations

import numpy as np

KEYPOINTS = ["pelvis", "neck", "left_knee", "left_foot", "right_knee", "right_foot",
             "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]


class PlanarHumanoid:  # robot segment lengths (independent copy)
    thigh = 0.40
    shin = 0.39
    torso = 0.48
    shoulder_x = 0.16
    shoulder_drop = 0.06
    upper_arm = 0.27
    forearm = 0.25


def terrain_height(x, verts):
    v = np.asarray(verts, dtype=float)
    return np.interp(x, v[:, 0], v[:, 1])


def terrain_sdf(P, verts):
    v = np.asarray(verts, dtype=float)
    P = np.asarray(P, dtype=float)
    A, B = v[:-1], v[1:]
    AB = B - A
    t = np.clip(np.sum((P[..., None, :] - A) * AB, axis=-1) / np.sum(AB * AB, axis=1), 0.0, 1.0)
    C = A + t[..., None] * AB
    d = np.sqrt(np.min(np.sum((P[..., None, :] - C) ** 2, axis=-1), axis=-1))
    inside = P[..., 1] < np.interp(P[..., 0], v[:, 0], v[:, 1])
    return np.where(inside, -d, d)


def box_sdf(P, center, size):
    q = np.abs(np.asarray(P, dtype=float) - np.asarray(center, dtype=float)) - 0.5 * np.asarray(size, dtype=float)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)

DT = 1.0 / 30.0
ROBOT = PlanarHumanoid()
EXPERIMENT_OF = {
    ("object_carry", False): "object_interaction_kinematics",
    ("terrain_step", False): "terrain_interaction_kinematics",
    ("object_carry", True): "augmented_interaction_kinematics",
    ("terrain_step", True): "augmented_interaction_kinematics",
}
HUMAN_LINKS = [(0, 1), (1, 6), (1, 9), (0, 2), (2, 3), (0, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)]


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


def _human_dims(rng: np.random.Generator, s: float) -> dict:
    j = lambda: float(rng.uniform(0.94, 1.06))  # noqa: E731
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


def _attempt(rng: np.random.Generator, kind: str, augmented: bool) -> tuple[dict, dict]:
    s = float(rng.uniform(0.90, 1.15))
    dims = _human_dims(rng, s)
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

    if kind == "terrain_step":
        W_r = float(rng.uniform(0.08, 0.16))
        lo, hi = x_r0 + 0.04, x_rf - W_r - 0.04
        if hi <= lo:
            raise Reject("no room for ramp")
        x_r = float(rng.uniform(lo, hi))
        H = float(rng.uniform(0.10, 0.24))
        x_c = x_rf + float(rng.uniform(0.12, 0.18))
        W_c = float(rng.uniform(0.14, 0.22))
        terrain_src = None  # column height filled in after the body is built
    else:
        terrain_src = flat

    tmp_terrain = flat if kind == "object_carry" else [
        [-4.0, 0.0], [x_r, 0.0], [x_r + W_r, H], [x_c, H], [x_c + 0.02, H + 2.0],
        [x_c + W_c - 0.02, H + 2.0], [x_c + W_c, H], [4.0, H]]
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

    # Target scene (augmentation of the source scene).
    terrain_tgt = [list(v) for v in terrain_src]
    obj_tgt = None if obj_src is None else {"size": list(obj_src["size"]), "center": obj_src["center"].copy()}
    aug_info = {}
    if augmented and kind == "object_carry":
        k = float(rng.uniform(0.85, 1.20))
        dpx = float(rng.choice([-1, 1]) * rng.uniform(0.03, 0.08))
        dpz = float(rng.uniform(-0.05, 0.05))
        tau = float(rng.uniform(1.0, 3.0))
        decay = np.exp(-t_idx * DT / tau)
        obj_tgt = {"size": [obj_src["size"][0] * k, obj_src["size"][1] * k],
                   "center": obj_src["center"] + np.stack([dpx * decay, dpz * decay], axis=1)}
        aug_info = {"size_scale": k, "offset": [dpx, dpz], "tau": tau}
    elif augmented and kind == "terrain_step":
        H2 = float(np.clip(H * rng.uniform(0.5, 1.6), 0.05, 0.32))
        if abs(H2 - H) < 0.04:
            raise Reject("augmentation too small")
        dH = H2 - H
        terrain_tgt = [[x, z + dH if (x > x_r + 1e-9) else z] for x, z in terrain_src]
        aug_info = {"platform_height": [H, H2]}

    # Desired robot contacts must be physically attainable in the target scene.
    for t in range(N):
        targets = {}
        if kind == "object_carry":
            cen, size = obj_tgt["center"][t], np.array(obj_tgt["size"])
            sc = size / np.array(obj_src["size"])
            rel_l = (lhand[t] - obj_src["center"][t]) * sc
            rel_r = (rhand[t] - obj_src["center"][t]) * sc
            targets = {"L": cen + rel_l, "R": cen + rel_r}
            if not _robot_can_hold(targets, px[t], 0.0, (cen, size)):
                raise Reject("robot cannot hold the box")
        elif hand_contact["right_hand"][t] == "terrain":
            x_h = rhand[t, 0]
            targets = {"R": np.array([x_h, float(terrain_height(x_h, terrain_tgt))])}
            ground = float(terrain_height(px[t], terrain_tgt))
            if not _robot_can_hold(targets, px[t], ground):
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
        "target_scene": scene(terrain_tgt, obj_tgt),
    }
    ann = {
        "experiment": EXPERIMENT_OF[(kind, augmented)],
        "stance": stance,
        "hand_contact": hand_contact,
        "human_scale": s,
        "augmented": bool(augmented),
        "augmentation": aug_info,
    }
    return inp, ann


def make_instance(master_seed: int, index: int, kind: str, augmented: bool, instance_id: str) -> tuple[dict, dict]:
    rng = np.random.default_rng([master_seed, index])
    for _ in range(500):
        try:
            inp, ann = _attempt(rng, kind, augmented)
        except Reject:
            continue
        inp = {"instance_id": instance_id, **inp}
        ann = {"instance_id": instance_id, **ann}
        return inp, ann
    raise RuntimeError(f"could not generate instance {instance_id}")


def make_split(master_seed: int, per_experiment: int, prefix: str) -> list[tuple[dict, dict]]:
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
        out.append(make_instance(master_seed, idx, kind, aug, iid))
        idx += 1
    return out


_TEST_MASTER_SEED = 0x5EC12E7_0B5C
TEST_PER_EXPERIMENT = 12


def test_split() -> list[tuple[dict, dict]]:
    return make_split(_TEST_MASTER_SEED, TEST_PER_EXPERIMENT, "test")
