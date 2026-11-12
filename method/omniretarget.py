"""OmniRetarget, scaled planar reproduction (paper Sec. III-A, Alg. 5, App. D).

For every frame t (warm-started from q*_{t-1}) we solve

    min_q  sum_i || L(p_i^source) - L(p_i^target(q)) ||^2 + || q - q_{t-1} ||_Q^2      (3a)
    s.t.   phi_j(q) >= 0             (non-penetration of robot links, 3b)
           q_min <= q <= q_max        (3c)
           v_min dt <= q - q_{t-1} <= v_max dt   (3d)
           p^F_t(q) = p^F_{t-1}       for source stance feet (3e)

with a sequential convex method: each iteration linearises the Laplacian
residuals (Gauss-Newton quadratic model), linearises the collision and
stance constraints around the current iterate, keeps the joint/velocity
bounds exact and adds the second-order-cone trust region ||dq||_2 <= 0.2
(15a-15g).  At most 10 iterations per frame.  The interaction mesh is the
Delaunay triangulation (2D analogue of the paper's tetrahedralisation) of
the source keypoints plus dense samples of the object surface and terrain;
uniform Laplacian weights; meshes are expressed in the object frame when an
object is present.  A foot is in stance when its source horizontal speed is
below 1 cm/s.

Source preprocessing follows the paper's direct scaling (App. A, Eq. 7/8):
the demonstration is scaled by alpha = h_robot / h_human (heights from the
demonstrator's bone lengths).  The whole demonstration -- human keypoints and
the source-scene samples they interact with -- is scaled about the ground
point below the pelvis in every frame, so the scaled demonstration stays
self-consistent while the walking path is kept; the target mesh uses the
robot and the actual target scene.

The convex sub-problem is solved exactly with SciPy's SLSQP (it is a small
QP with one quadratic trust-region constraint); an L-infinity elastic slack
on the linearised collision rows keeps it feasible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

from data.planar_robot import (KEYPOINTS, NAME_TO_IDX, Q_MAX, Q_MIN, PlanarHumanoid,
                               clip_to_limits, link_sample_weights, per_frame_velocity_bounds)
from data.scene_geometry import box_perimeter_uv, box_sdf, terrain_sdf

ROBOT = PlanarHumanoid()
FEET = [NAME_TO_IDX["left_foot"], NAME_TO_IDX["right_foot"]]
STANCE_SPEED = 0.01        # m/s (Sec. III-A)
TRUST_RADIUS = 0.2         # epsilon in (15g)
MAX_SQP_ITERS = 10         # App. D
FIRST_FRAME_ITERS = 40     # frame 0 has no warm start from a previous solution
Q_WEIGHTS = np.array([0.5, 0.5] + [0.05] * 9)
ELASTIC_PENALTY = 1e3
COLLISION_MARGIN = 0.25    # only linearise collision rows of nearby link samples
N_LINK_SAMPLES = 5
W_LINK = link_sample_weights(N_LINK_SAMPLES)
TERRAIN_SPACING = 0.05
BOX_UV = box_perimeter_uv(8)


# ---------------------------------------------------------------------------
# Scene sampling with source/target correspondence
# ---------------------------------------------------------------------------

def terrain_samples(src_verts, tgt_verts, x_lo, x_hi):
    """Corresponding samples along the source and target terrain polylines."""
    S = np.asarray(src_verts, dtype=float)
    T = np.asarray(tgt_verts, dtype=float)
    ps, pt = [], []
    for k in range(len(S) - 1):
        a, b = S[k], S[k + 1]
        c, d = T[k], T[k + 1]
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / TERRAIN_SPACING)))
        u = np.arange(n) / n
        seg_s = a + u[:, None] * (b - a)
        seg_t = c + u[:, None] * (d - c)
        keep = (seg_s[:, 0] >= x_lo) & (seg_s[:, 0] <= x_hi)
        ps.append(seg_s[keep])
        pt.append(seg_t[keep])
    ps.append(S[-1:][(S[-1:, 0] >= x_lo) & (S[-1:, 0] <= x_hi)])
    pt.append(T[-1:][(S[-1:, 0] >= x_lo) & (S[-1:, 0] <= x_hi)])
    return np.concatenate(ps), np.concatenate(pt)


def neighbour_laplacian(points: np.ndarray) -> np.ndarray:
    """Uniform-weight Laplacian operator L = I - W from a Delaunay mesh."""
    n = len(points)
    tri = Delaunay(points, qhull_options="QJ")
    A = np.zeros((n, n))
    for simplex in tri.simplices:
        for i in simplex:
            for j in simplex:
                if i != j:
                    A[i, j] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    W = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)
    return np.eye(n) - W


# ---------------------------------------------------------------------------
# Per-frame problem data
# ---------------------------------------------------------------------------

def human_height(src0: np.ndarray) -> float:
    P = lambda n: src0[NAME_TO_IDX[n]]  # noqa: E731
    torso = np.linalg.norm(P("neck") - P("pelvis"))
    thigh = 0.5 * (np.linalg.norm(P("left_knee") - P("pelvis")) + np.linalg.norm(P("right_knee") - P("pelvis")))
    shin = 0.5 * (np.linalg.norm(P("left_foot") - P("left_knee")) + np.linalg.norm(P("right_foot") - P("right_knee")))
    return float(torso + thigh + shin)


class FrameProblem:
    def __init__(self, task: dict, t: int, terr_s: np.ndarray, terr_t: np.ndarray, objective: str,
                 alpha: float = 1.0, anchor: np.ndarray | None = None):
        self.t = t
        raw = np.asarray(task["source_keypoints"][t], dtype=float)
        anchor = np.zeros(2) if anchor is None else anchor

        def scale(p):  # direct scaling of the demonstration (Eq. 7/8)
            return anchor + alpha * (np.asarray(p, dtype=float) - anchor)

        src = scale(raw)
        terr_s = scale(terr_s)
        self.src = src
        self.objective = objective
        tgt_scene = task["target_scene"]
        src_scene = task["source_scene"]
        self.terrain_tgt = np.asarray(tgt_scene["terrain"], dtype=float)
        self.box_tgt = None
        scene_s = [terr_s]
        scene_t = [terr_t]
        origin_s = np.zeros(2)
        origin_t = np.zeros(2)
        if src_scene.get("object"):
            cs = np.asarray(src_scene["object"]["center"][t], dtype=float)
            ss = np.asarray(src_scene["object"]["size"], dtype=float)
            ct = np.asarray(tgt_scene["object"]["center"][t], dtype=float)
            st = np.asarray(tgt_scene["object"]["size"], dtype=float)
            self.box_tgt = (ct, st)
            scene_s.insert(0, scale(cs + BOX_UV * ss))
            scene_t.insert(0, ct + BOX_UV * st)
            origin_s, origin_t = scale(cs), ct  # object-frame interaction mesh (App. C-2)
        self.scene_t = np.concatenate(scene_t) - origin_t
        self.origin_t = origin_t
        mesh_s = np.concatenate([src] + scene_s) - origin_s
        if objective == "mesh":
            self.L = neighbour_laplacian(mesh_s)
            self.L_src = self.L @ mesh_s
            self.L_body = self.L[:, :len(src)]
            self.L_scene_term = self.L[:, len(src):] @ self.scene_t

    def residual(self, fk: np.ndarray, Jfk: np.ndarray):
        if self.objective == "mesh":
            r = self.L_body @ (fk - self.origin_t) + self.L_scene_term - self.L_src
            J = np.einsum("vk,kdj->vdj", self.L_body, Jfk)
            return r.ravel(), J.reshape(-1, Jfk.shape[-1])
        # Ablation: GMR-style keypoint position matching in place of the mesh.
        r = fk - self.src
        return r.ravel(), Jfk.reshape(-1, Jfk.shape[-1])

    def collision_rows(self, fk: np.ndarray, Jfk: np.ndarray):
        S = W_LINK @ fk
        JS = np.einsum("sk,kdj->sdj", W_LINK, Jfk)
        rows, vals = [], []
        phi, grad = terrain_sdf(S, self.terrain_tgt, with_grad=True)
        keep = phi < COLLISION_MARGIN
        rows.append(np.einsum("sd,sdj->sj", grad[keep], JS[keep]))
        vals.append(phi[keep])
        if self.box_tgt is not None:
            phi, grad = box_sdf(S, self.box_tgt[0], self.box_tgt[1], with_grad=True)
            keep = phi < COLLISION_MARGIN
            rows.append(np.einsum("sd,sdj->sj", grad[keep], JS[keep]))
            vals.append(phi[keep])
        return np.concatenate(rows), np.concatenate(vals)


def solve_subproblem(J, r, qbar, qref, qw, G, phi, Aeq, beq, lo, hi, trust):
    n = len(qbar)
    JtJ = J.T @ J
    Jtr = J.T @ r

    def f(z):
        dq = z[:n]
        e = qbar + dq - qref
        return float(dq @ JtJ @ dq + 2.0 * Jtr @ dq + e @ (qw * e) + ELASTIC_PENALTY * z[n])

    def g(z):
        dq = z[:n]
        e = qbar + dq - qref
        out = np.empty(n + 1)
        out[:n] = 2.0 * (JtJ @ dq) + 2.0 * Jtr + 2.0 * qw * e
        out[n] = ELASTIC_PENALTY
        return out

    cons = [{"type": "ineq",
             "fun": lambda z: np.array([trust * trust - z[:n] @ z[:n]]),
             "jac": lambda z: np.concatenate([-2.0 * z[:n], [0.0]])[None, :]}]
    if len(G):
        Gz = np.hstack([G, np.ones((len(G), 1))])
        cons.append({"type": "ineq", "fun": lambda z: Gz @ z + phi, "jac": lambda z: Gz})
    if Aeq is not None and len(Aeq):
        Az = np.hstack([Aeq, np.zeros((len(Aeq), 1))])
        cons.append({"type": "eq", "fun": lambda z: Az @ z + beq, "jac": lambda z: Az})
    s0 = float(max(0.0, -np.min(phi))) if len(phi) else 0.0
    z0 = np.zeros(n + 1)
    z0[n] = s0
    bounds = [(float(a), float(b)) for a, b in zip(lo, hi)] + [(0.0, None)]
    try:
        res = minimize(f, z0, jac=g, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-10})
        z = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(z)):
            return np.zeros(n)
        return np.clip(z[:n], lo, hi)
    except Exception:  # pragma: no cover - numerical safety net
        return np.zeros(n)


# ---------------------------------------------------------------------------
# Initial guess for frame 0 (copy source limb angles, then refine with IK)
# ---------------------------------------------------------------------------

def _angle(v):
    return float(np.arctan2(v[0], -v[1]))


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def initial_configuration(src: np.ndarray, terrain) -> np.ndarray:
    P = {n: src[i] for i, n in enumerate(KEYPOINTS)}
    lh = _angle(P["left_knee"] - P["pelvis"])
    rh = _angle(P["right_knee"] - P["pelvis"])
    ls = _angle(P["left_elbow"] - P["left_shoulder"])
    rs = _angle(P["right_elbow"] - P["right_shoulder"])
    torso_vec = P["neck"] - P["pelvis"]
    q = np.array([
        P["pelvis"][0], P["pelvis"][1], float(np.arctan2(torso_vec[0], torso_vec[1])),
        lh, _wrap(_angle(P["left_foot"] - P["left_knee"]) - lh),
        rh, _wrap(_angle(P["right_foot"] - P["right_knee"]) - rh),
        ls, _wrap(_angle(P["left_hand"] - P["left_elbow"]) - ls),
        rs, _wrap(_angle(P["right_hand"] - P["right_elbow"]) - rs),
    ])
    q = np.clip(q, Q_MIN, Q_MAX)
    from scipy.optimize import minimize as _min

    def obj(qq):
        return float(np.sum((ROBOT.fk(qq) - src) ** 2))

    res = _min(obj, q, method="L-BFGS-B", bounds=list(zip(Q_MIN, Q_MAX)), options={"maxiter": 200})
    return np.clip(np.asarray(res.x, dtype=float), Q_MIN, Q_MAX)


# ---------------------------------------------------------------------------
# Trajectory retargeting
# ---------------------------------------------------------------------------

def retarget(task: dict, objective: str = "mesh", extra_residual=None,
             max_iters: int = MAX_SQP_ITERS, trust: float = TRUST_RADIUS) -> np.ndarray:
    """Retarget one task; returns q of shape (num_frames, 11).

    ``extra_residual(t, fk, Jfk) -> (r, J) | None`` may append task-specific
    least-squares costs to the objective (App. B-5: the core objective can be
    augmented with task-specific terms).
    """
    src_all = np.asarray(task["source_keypoints"], dtype=float)
    N = len(src_all)
    dt = float(task["dt"])
    vstep = per_frame_velocity_bounds(dt)
    px = src_all[:, 0, 0]
    terr_s, terr_t = terrain_samples(task["source_scene"]["terrain"], task["target_scene"]["terrain"],
                                     float(px.min()) - 0.9, float(px.max()) + 0.9)
    alpha = ROBOT.height / human_height(src_all[0])
    src_terrain = np.asarray(task["source_scene"]["terrain"], dtype=float)
    anchors = np.stack([px, np.interp(px, src_terrain[:, 0], src_terrain[:, 1])], axis=1)
    q_out = np.zeros((N, len(Q_MIN)))
    q_prev = None
    feet_prev = None
    for t in range(N):
        prob = FrameProblem(task, t, terr_s, terr_t, objective, alpha, anchors[t])
        if q_prev is None:
            qbar = initial_configuration(prob.src, task["target_scene"]["terrain"])
            qref, qw, iters = qbar.copy(), 0.1 * Q_WEIGHTS, FIRST_FRAME_ITERS
            box_lo, box_hi = Q_MIN.copy(), Q_MAX.copy()
            stance = []
        else:
            qbar = q_prev.copy()
            qref, qw, iters = q_prev, Q_WEIGHTS, max_iters
            box_lo = np.maximum(Q_MIN, q_prev - vstep)
            box_hi = np.minimum(Q_MAX, q_prev + vstep)
            stance = [f for f in FEET if abs(src_all[t, f, 0] - src_all[t - 1, f, 0]) / dt < STANCE_SPEED]
        for _ in range(iters):
            fk, Jfk = ROBOT.fk_jacobian(qbar)
            r, J = prob.residual(fk, Jfk)
            if extra_residual is not None:
                ex = extra_residual(t, fk, Jfk)
                if ex is not None:
                    r = np.concatenate([r, ex[0]])
                    J = np.vstack([J, ex[1]])
            G, phi = prob.collision_rows(fk, Jfk)
            if stance:
                Aeq = np.concatenate([Jfk[f] for f in stance])
                beq = np.concatenate([fk[f] - feet_prev[f] for f in stance])
            else:
                Aeq, beq = None, None
            dq = solve_subproblem(J, r, qbar, qref, qw, G, phi, Aeq, beq,
                                  box_lo - qbar, box_hi - qbar, trust)
            qbar = np.clip(qbar + dq, box_lo, box_hi)
            if np.linalg.norm(dq) < 1e-6:
                break
        q_t = clip_to_limits(qbar, q_prev, dt)
        q_out[t] = q_t
        q_prev = q_t
        feet_prev = ROBOT.fk(q_t)
    return q_out


def run_dir(inputs: Path, outputs: Path, method_name: str, **kwargs) -> None:
    outputs.mkdir(parents=True, exist_ok=True)
    for path in sorted(inputs.glob("*.json")):
        task = json.loads(path.read_text())
        q = retarget(task, **kwargs)
        out = {"instance_id": task["instance_id"], "method": method_name, "q": np.round(q, 10).tolist()}
        (outputs / f"{task['instance_id']}.json").write_text(json.dumps(out))
        print(f"[{method_name}] {task['instance_id']} done", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True, help="directory of task input JSON files")
    parser.add_argument("--outputs", type=Path, required=True, help="directory for per-instance outputs")
    parser.add_argument("--objective", choices=["mesh", "keypoint"], default="mesh",
                        help="'keypoint' is the no-interaction-mesh ablation")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    name = args.name or ("omniretarget" if args.objective == "mesh" else "omniretarget_no_mesh")
    run_dir(args.inputs, args.outputs, name, objective=args.objective)


if __name__ == "__main__":
    main()
