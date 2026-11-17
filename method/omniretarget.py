"""OmniRetarget (paper Sec. III, App. A/B-5/C/D): interaction-mesh retargeting with
hard kinematic constraints, solved frame by frame with sequential SOCPs.

For every frame t:
  * the interaction mesh is the Delaunay tetrahedralisation of the source
    vertices: the demonstrator's 15 key joints (direct scaling by alpha, App. A)
    plus points sampled on the manipulated object / terrain (Sec. III-A); for the
    box-carrying task the mesh is built in the object's frame (App. C-2);
  * the robot configuration minimises the Laplacian deformation between the
    source mesh and the robot mesh (Eq. 1-2, uniform weights) plus temporal
    smoothness (Eq. 3a), subject to non-penetration (3b), joint limits (3c),
    joint velocity limits (3d) and stance-foot sticking (3e);
  * (3) is solved by an SQP-style loop (App. D): up to 10 iterations (50 at the
    first frame), each an SOCP with the objective quadratised and the
    constraints linearised around the current iterate and a trust region
    ||dq|| <= 0.2 (15g), warm-started from the previous frame.
Augmented scenes (Sec. III-B) are solved after the nominal scene, anchoring
the base and legs to the nominal solution (Eq. 4) and starting from its first
frame so that the initial feet match (Eq. 5); with terrain augmentation a grid
of ground points joins the interaction mesh.

The released implementation (holosoma_retargeting, Apache-2.0) was used as the
reference for weights and details (Laplacian weight 10, smoothness 0.2, waist
costs and bounds, 1 mm penetration and foot-sticking tolerances (sticking in 3D as in the paper's (3e)), stance from
1 cm/frame source toe speed, 100 surface samples, nominal tracking 5 e^{-t/10}).
SOCPs are solved with Clarabel; robot kinematics, Jacobians and signed
distances come from MuJoCo.

Usage: python method/omniretarget.py --inputs DIR --outputs DIR [--objective mesh|keypoint]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clarabel  # noqa: E402
import mujoco  # noqa: E402

from data.scene import build_scene_model, quat_to_mat, set_frame  # noqa: E402

# Keypoint correspondences (paper: "semantically consistent correspondence"), from the released code.
SMPLH_MAP = {
    "Pelvis": "pelvis_contour_link", "L_Hip": "left_hip_pitch_link", "R_Hip": "right_hip_pitch_link",
    "L_Knee": "left_knee_link", "R_Knee": "right_knee_link", "L_Shoulder": "left_shoulder_roll_link",
    "R_Shoulder": "right_shoulder_roll_link", "L_Elbow": "left_elbow_link", "R_Elbow": "right_elbow_link",
    "L_Ankle": "left_ankle_intermediate_1_link", "R_Ankle": "right_ankle_intermediate_1_link",
    "L_Toe": "left_ankle_roll_sphere_5_link", "R_Toe": "right_ankle_roll_sphere_5_link",
    "L_Wrist": "left_sphere_hand_link", "R_Wrist": "right_sphere_hand_link",
}
MOCAP_MAP = {
    "Spine1": "pelvis_contour_link", "LeftUpLeg": "left_hip_pitch_link", "LeftLeg": "left_knee_link",
    "LeftToeBase": "left_ankle_roll_sphere_5_link", "RightUpLeg": "right_hip_pitch_link",
    "RightLeg": "right_knee_link", "RightToeBase": "right_ankle_roll_sphere_5_link",
    "LeftArm": "left_shoulder_roll_link", "LeftForeArm": "left_elbow_link",
    "LeftHandMiddle3": "left_sphere_hand_link", "RightArm": "right_shoulder_roll_link",
    "RightForeArm": "right_elbow_link", "RightHandMiddle3": "right_sphere_hand_link",
    "LeftFoot": "left_ankle_intermediate_1_link", "RightFoot": "right_ankle_intermediate_1_link",
}
TOES = {"smplh": ("L_Toe", "R_Toe"), "mocap": ("LeftToeBase", "RightToeBase")}
HIPS = {"smplh": ("L_Hip", "R_Hip"), "mocap": ("LeftUpLeg", "RightUpLeg")}
FOOT_STICK_LINKS = {
    "left": [f"left_ankle_roll_sphere_{k}_link" for k in (1, 2, 3, 4)],
    "right": [f"right_ankle_roll_sphere_{k}_link" for k in (1, 2, 3, 4)],
}

# Solver settings (released code defaults).
LAPLACIAN_WEIGHT = 10.0
SMOOTH_WEIGHT = 0.2
TRUST_REGION = 0.2
COLLISION_THRESHOLD = 0.1
PENETRATION_TOL = 1e-3
FOOT_STICK_TOL = 1e-3
STANCE_SPEED = 0.01           # m per frame, scaled source toe (released code)
ITERS_FIRST, ITERS = 50, 10
N_SURFACE_SAMPLES = 100
TOP_SURFACE_WEIGHT = 20.0     # terrain: sample top faces 20x more densely (released code)
GROUND_GRID = (8, 2.0)        # terrain augmentation: 8x8 ground points over [-2, 2]^2
W_NOMINAL, TAU_NOMINAL = 5.0, 10.0
# (3d) applied to the floating base too: per-frame bound on the base velocity (tangent coordinates),
# twice the fastest demonstrated pelvis motion (1.5 m/s at robot scale).
BASE_V_MAX = np.array([3.0, 3.0, 3.0, 6.0, 6.0, 6.0])   # m/s (world xyz), rad/s (local axes)
# Joint-index (0..28, MJCF order) extra bounds and costs from the released code (waist, elbows, wrists).
MANUAL_LB = {13: -0.3, 14: -0.1, 19: -0.1, 20: -0.1, 21: -0.05, 26: -0.1, 27: -0.1, 28: -0.05}
MANUAL_UB = {13: 0.3, 18: 1.4, 19: 0.2, 20: 0.3, 21: 0.05, 25: 1.4, 26: 0.2, 27: 0.3, 28: 0.05}
MANUAL_COST = {12: 0.2, 13: 0.2}          # waist yaw, waist roll
NOMINAL_DOFS = np.arange(18)              # floating base (6) + legs (12), velocity coordinates
KEYPOINT_WEIGHT = 10.0                    # ablation objective (no interaction mesh)


def box_surface_params(n: int, half: np.ndarray, rng: np.random.Generator, top_weight: float = 1.0) -> np.ndarray:
    """Normalised surface points u in [-1, 1]^3 of a box, area-weighted (top face optionally heavier)."""
    hx, hy, hz = half
    faces = []  # (axis, sign, area)
    for ax, (a, b) in enumerate([(hy, hz), (hx, hz), (hx, hy)]):
        for sgn in (-1, 1):
            w = top_weight if (ax == 2 and sgn == 1) else 1.0
            faces.append((ax, sgn, 4 * a * b * w))
    p = np.array([f[2] for f in faces])
    idx = rng.choice(len(faces), size=n, p=p / p.sum())
    u = rng.uniform(-1, 1, size=(n, 3))
    for k, (ax, sgn, _) in enumerate(faces):
        u[idx == k, ax] = sgn
    return u


def support_height(xy: np.ndarray, scene: dict) -> np.ndarray:
    """Height of the highest static surface (ground or terrain box top) below points xy (..., 2)."""
    h = np.full(xy.shape[:-1], float(scene.get("ground_z", 0.0)))
    for b in scene["boxes"]:
        if "pose" not in b:
            continue
        p = np.asarray(b["pose"], dtype=float)
        R = quat_to_mat(p[3:7])
        half = np.asarray(b["half_size"], dtype=float)
        loc = (xy - p[:2]) @ R[:2, :2]
        inside = np.all(np.abs(loc) <= half[:2], axis=-1)
        top = p[2] + np.abs(R[2]) @ half
        h = np.where(inside, np.maximum(h, top), h)
    return h


def laplacian_from_tets(tets: np.ndarray, n: int) -> np.ndarray:
    """Uniform-weight graph Laplacian (Eq. 1) of the tetrahedral mesh: L = I - D^-1 A."""
    e = np.concatenate([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float).toarray()
    deg = A.sum(1)
    L = np.eye(n)
    nz = deg > 0
    L[nz] -= A[nz] / deg[nz, None]
    return L


class Retargeter:
    def __init__(self, inst: dict, scene: dict, objective: str = "mesh", contact_weight: float = 0.0):
        self.inst = inst
        self.scene = scene
        self.objective = objective
        self.contact_weight = contact_weight
        self.T = int(inst["num_frames"])
        self.dt = 1.0 / float(inst["fps"])
        self.alpha = float(inst["alpha"])
        self.task = inst["task"]
        names = inst["joint_names"]
        self.fmt = "smplh" if "L_Wrist" in names else "mocap"
        kmap = SMPLH_MAP if self.fmt == "smplh" else MOCAP_MAP
        self.src_idx = [names.index(k) for k in kmap]
        self.model, self.data, self.mocap = build_scene_model(scene)
        m = self.model
        m.geom_margin[:] = COLLISION_THRESHOLD
        self.kp_bodies = [m.body(b).id for b in kmap.values()]
        self.foot_bodies = {s: [m.body(b).id for b in v] for s, v in FOOT_STICK_LINKS.items()}
        self.robot_geoms = np.array([g for g in range(m.ngeom) if m.geom_contype[g] == 1])
        self.nv = m.nv
        # joint limits (actuated) with the released code's manual bounds
        lo, hi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()
        for k, v in MANUAL_LB.items():
            lo[k] = max(lo[k], v)
        for k, v in MANUAL_UB.items():
            hi[k] = min(hi[k], v)
        self.q_lo, self.q_hi = lo, hi
        lim = json.loads((ROOT / "data/robot/g1_limits.json").read_text())
        self.v_max = np.array([j["velocity"] for j in lim["joints"]])
        self.cost_diag = np.zeros(29)
        for k, v in MANUAL_COST.items():
            self.cost_diag[k] = v
        self._prepare_source()

    # ------------------------------------------------------------------ source
    def _prepare_source(self):
        inst = self.inst
        J = np.asarray(inst["human_joints"], dtype=float)
        names = inst["joint_names"]
        toes = [names.index(t) for t in TOES[self.fmt]]
        # Released code: lower the demonstrator so its lowest toe touches the floor.  A clip
        # window may never touch the floor (standing on a platform), so the toe height is
        # measured above the supporting surface of the source scene instead.
        support = support_height(J[:, toes, :2], inst["source_scene"])
        z_min = float(np.min(J[:, toes, 2] - support))
        if z_min >= 0.1:                        # demonstrator on a mat (released code)
            z_min -= 0.1
        J = J.copy()
        J[:, :, 2] -= z_min
        self.src = self.alpha * J                                   # direct scaling (Eq. 7/8)
        self.src_kp = self.src[:, self.src_idx]
        toe_xy = self.src[:, toes, :2]
        v = np.linalg.norm(np.diff(toe_xy, axis=0), axis=2)
        stance = np.vstack([np.zeros((1, 2), bool), v <= STANCE_SPEED])
        self.stance = {"left": stance[:, 0], "right": stance[:, 1]}
        hips = [names.index(h) for h in HIPS[self.fmt]]
        left = self.src[0, hips[0]] - self.src[0, hips[1]]
        fwd = np.cross(left, [0.0, 0.0, 1.0])
        self.yaw0 = float(np.arctan2(fwd[1], fwd[0]))

        rng = np.random.default_rng(0)
        nominal = inst["nominal_scene"]
        if self.task == "object":
            nb, tb = nominal["boxes"][0], self.scene["boxes"][0]
            u = box_surface_params(N_SURFACE_SAMPLES, np.asarray(nb["half_size"]), rng)
            self.obj_src_local = self.alpha * (u * np.asarray(nb["half_size"]) + np.asarray(nb["offset"]))
            self.obj_tgt_local = u * np.asarray(tb["half_size"]) + np.asarray(tb["offset"])
            Ps, Pt = np.asarray(nb["poses"], dtype=float), np.asarray(tb["poses"], dtype=float)
            self.src_frame = (Ps[:, :3], quat_to_mat(Ps[:, 3:7]))
            self.tgt_frame = (Pt[:, :3], quat_to_mat(Pt[:, 3:7]))
        else:
            src_pts, tgt_pts = [], []
            for nb, tb in zip(nominal["boxes"], self.scene["boxes"]):
                share = max(8, int(round(N_SURFACE_SAMPLES / len(nominal["boxes"]))))
                u = box_surface_params(share, np.asarray(nb["half_size"]), rng, TOP_SURFACE_WEIGHT)
                for b, out in ((nb, src_pts), (tb, tgt_pts)):
                    p = np.asarray(b["pose"], dtype=float)
                    R = quat_to_mat(p[3:7])
                    out.append((u * np.asarray(b["half_size"])) @ R.T + p[:3])
            src_pts, tgt_pts = np.vstack(src_pts), np.vstack(tgt_pts)
            if inst.get("augmentation"):
                n, r = GROUND_GRID
                g = np.linspace(-r, r, n)
                gx, gy = np.meshgrid(g, g)
                grid = np.stack([gx.ravel(), gy.ravel(), np.zeros(n * n)], 1)
                src_pts, tgt_pts = np.vstack([src_pts, grid]), np.vstack([tgt_pts, grid])
            self.env_src, self.env_tgt = src_pts, tgt_pts

    # ------------------------------------------------------------ kinematics
    def _fk(self, q: np.ndarray, t: int):
        d = self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def _point_jac(self, body: int, point: np.ndarray) -> np.ndarray:
        jp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.model, self.data, jp, None, point, body)
        return jp

    def _keypoints(self):
        P = np.array([self.data.xpos[b] for b in self.kp_bodies])
        J = np.stack([self._point_jac(b, self.data.xpos[b]) for b in self.kp_bodies])
        return P, J

    def _collisions(self):
        """Linearised signed distances (phi, n^T J) for robot-scene pairs closer than the threshold."""
        m, d = self.model, self.data
        mujoco.mj_collision(m, d)
        pairs = set()
        for k in range(d.ncon):
            c = d.contact[k]
            pairs.add((int(c.geom1), int(c.geom2)))
        phis, rows = [], []
        fromto = np.zeros(6)
        for g1, g2 in pairs:
            if m.geom_contype[g1] == 1:
                gr, gs = g1, g2
            else:
                gr, gs = g2, g1
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, gr, gs, COLLISION_THRESHOLD, fromto)
            if dist > COLLISION_THRESHOLD:
                continue
            # The witness-point order of mj_geomDistance is not reliable for mesh geoms, so the
            # robot point is the one off the scene surface and the normal is the scene geom's
            # signed-distance gradient there (scene geoms are boxes and the ground plane).
            pa, pb = fromto[:3].copy(), fromto[3:].copy()
            sa, na = self._scene_sdf_grad(gs, pa)
            sb, nb = self._scene_sdf_grad(gs, pb)
            pr, n = (pa, na) if abs(sa - dist) <= abs(sb - dist) else (pb, nb)
            Jr = self._point_jac(int(m.geom_bodyid[gr]), pr)
            phis.append(dist)
            rows.append(n @ Jr)
        if not rows:
            return np.zeros(0), np.zeros((0, self.nv))
        return np.array(phis), np.array(rows)

    def _scene_sdf_grad(self, g: int, p: np.ndarray) -> tuple[float, np.ndarray]:
        """Signed distance and outward gradient of scene geom g (plane or box) at world point p."""
        m, d = self.model, self.data
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            return float(p[2] - d.geom_xpos[g][2]), np.array([0.0, 0.0, 1.0])
        R = d.geom_xmat[g].reshape(3, 3)
        loc = R.T @ (p - d.geom_xpos[g])
        q = np.abs(loc) - m.geom_size[g]
        out = np.maximum(q, 0.0)
        if np.any(q > 0):
            s = float(np.linalg.norm(out))
            gl = np.sign(loc) * out / max(s, 1e-12)
        else:
            k = int(np.argmax(q))
            s = float(q[k])
            gl = np.zeros(3)
            gl[k] = 1.0 if loc[k] >= 0 else -1.0
        return s, R @ gl

    # ----------------------------------------------------------------- mesh
    def _frame_mesh(self, t: int):
        """Source vertices (frame-specific) and the mesh Laplacian."""
        if self.task == "object":
            c, R = self.src_frame[0][t], self.src_frame[1][t]
            kp = (self.src_kp[t] - c) @ R            # into the object frame
            V = np.vstack([kp, self.obj_src_local])
        else:
            V = np.vstack([self.src_kp[t], self.env_src])
        try:
            tets = Delaunay(V).simplices
        except Exception:  # degenerate (e.g. coplanar) vertex set: joggled input
            tets = Delaunay(V, qhull_options="QJ").simplices
        L = laplacian_from_tets(tets, len(V))
        return V, L

    # ---------------------------------------------------------------- solve
    def _qp(self, P, qvec, eq, ineq, trust: bool):
        """min 1/2 x'Px + q'x  s.t. Aeq x = beq, G x <= h, ||x|| <= trust (Clarabel)."""
        n = self.nv
        A_blocks, b_blocks, cones = [], [], []
        if eq is not None and len(eq[1]):
            A_blocks.append(eq[0]); b_blocks.append(eq[1]); cones.append(clarabel.ZeroConeT(len(eq[1])))
        if ineq is not None and len(ineq[1]):
            A_blocks.append(ineq[0]); b_blocks.append(ineq[1]); cones.append(clarabel.NonnegativeConeT(len(ineq[1])))
        if trust:
            A_blocks.append(np.vstack([np.zeros((1, n)), -np.eye(n)]))
            b_blocks.append(np.concatenate([[TRUST_REGION], np.zeros(n)]))
            cones.append(clarabel.SecondOrderConeT(n + 1))
        A = sparse.csc_matrix(np.vstack(A_blocks))
        b = np.concatenate(b_blocks)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 100
        try:
            sol = clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P)), qvec, A, b, cones, settings).solve()
        except Exception:
            return None, np.inf
        ok = str(sol.status) in ("Solved", "AlmostSolved") and np.all(np.isfinite(sol.x))
        return (np.array(sol.x), float(sol.obj_val)) if ok else (None, np.inf)

    def _iteration(self, q, q_prev, t, V_src, L, foot_ref, q_nom, w_nom, init_feet):
        m, nv = self.model, self.nv
        self._fk(q, t)
        P_kp, J_kp = self._keypoints()
        if self.task == "object":
            c, R = self.tgt_frame[0][t], self.tgt_frame[1][t]
            P_kp = (P_kp - c) @ R
            J_kp = np.einsum("ji,kjv->kiv", R, J_kp)
            V_tgt = np.vstack([P_kp, self.obj_tgt_local])
        else:
            V_tgt = np.vstack([P_kp, self.env_tgt])
        K = len(self.kp_bodies)
        H = np.zeros((nv, nv))
        g = np.zeros(nv)
        if self.objective == "mesh":
            lap_tgt = L @ V_tgt
            lap_src = L @ V_src
            M = (L[:, :K] @ J_kp.reshape(K, 3 * nv)).reshape(len(V_tgt), 3, nv).reshape(-1, nv)
            r = (lap_src - lap_tgt).reshape(-1)
            H += LAPLACIAN_WEIGHT * M.T @ M
            g += -LAPLACIAN_WEIGHT * M.T @ r
            cost0 = LAPLACIAN_WEIGHT * float(r @ r)
        else:  # ablation: match the scaled keypoints (in the same frame) instead of the mesh
            M = J_kp.reshape(-1, nv)
            r = (V_src[:K] - V_tgt[:K]).reshape(-1)
            H += KEYPOINT_WEIGHT * M.T @ M
            g += -KEYPOINT_WEIGHT * M.T @ r
            cost0 = KEYPOINT_WEIGHT * float(r @ r)
        # smoothness ||q + dq - q_prev||^2 in tangent coordinates
        dprev = np.zeros(nv)
        mujoco.mj_differentiatePos(m, dprev, 1.0, q, q_prev)
        H += SMOOTH_WEIGHT * np.eye(nv)
        g += -SMOOTH_WEIGHT * dprev
        # waist costs on the absolute joint value
        cd = np.concatenate([np.zeros(6), self.cost_diag])
        H += np.diag(cd)
        g += cd * np.concatenate([np.zeros(6), q[7:]])
        if q_nom is not None and w_nom > 0:
            dn = np.zeros(nv)
            mujoco.mj_differentiatePos(m, dn, 1.0, q, q_nom)
            w = np.zeros(nv)
            w[NOMINAL_DOFS] = w_nom
            H += np.diag(w)
            g += -w * dn

        if self.contact_weight > 0:
            self._add_contact_cost(t, H, g)
        H += 1e-9 * np.eye(nv)
        G_rows, h_rows = [], []
        # (3b) non-penetration: phi + J dq >= -tol
        phi, Jn = self._collisions()
        if len(phi):
            G_rows.append(-Jn); h_rows.append(phi + PENETRATION_TOL)
        # (3e) stance-foot sticking p_t = p_{t-1} (within +-tol).  The released code pins only x, y;
        # the paper's (3e) pins the full position, which also keeps planted feet from leaving the floor.
        for side, ref in foot_ref.items():
            for b, p_ref in zip(self.foot_bodies[side], ref):
                p = self.data.xpos[b]
                Jp = self._point_jac(b, p)
                G_rows += [Jp, -Jp]
                h_rows += [p_ref - p + FOOT_STICK_TOL, -(p_ref - p - FOOT_STICK_TOL)]
        # (5) augmented scenes: initial feet at the nominal solution's
        if init_feet is not None:
            for side in ("left", "right"):
                for b, p_ref in zip(self.foot_bodies[side], init_feet[side]):
                    p = self.data.xpos[b]
                    Jp = self._point_jac(b, p)
                    G_rows += [Jp, -Jp]
                    h_rows += [p_ref - p + FOOT_STICK_TOL, -(p_ref - p - FOOT_STICK_TOL)]
        # (3c) joint limits; (3d) velocity limits on the joints and the floating base (not at the
        # first frame, which has no predecessor)
        qa, qpa = q[7:], q_prev[7:]
        first = t == 0
        vj = np.inf if first else self.v_max * self.dt
        lo = np.maximum(self.q_lo, qpa - vj) - qa
        hi = np.minimum(self.q_hi, qpa + vj) - qa
        if not first:
            vb = BASE_V_MAX * self.dt
            lo = np.concatenate([dprev[:6] - vb, lo])
            hi = np.concatenate([dprev[:6] + vb, hi])
            S = np.eye(nv)
        else:
            S = np.hstack([np.zeros((29, 6)), np.eye(29)])
        lo = np.minimum(lo, 0.0)  # keep dq = 0 feasible if the iterate is (numerically) outside
        hi = np.maximum(hi, 0.0)
        G_hard, h_hard = np.vstack([S, -S]), np.concatenate([hi, -lo])
        G_soft = np.vstack(G_rows) if G_rows else np.zeros((0, nv))
        h_soft = np.concatenate(h_rows) if h_rows else np.zeros(0)
        G, h = np.vstack([G_soft, G_hard]), np.concatenate([h_soft, h_hard])
        dq, val = self._qp(2 * H, 2 * g, None, (G, h), trust=True)
        if dq is None:
            # The linearised hard constraints can be mutually infeasible on real data (the released
            # code raises an error).  Fallback: penalised slack on the collision and foot rows only;
            # joint position/velocity limits stay hard.
            dq = self._elastic(H, g, G_soft, h_soft, G_hard, h_hard)
        q_new = q.copy()
        mujoco.mj_integratePos(m, q_new, dq, 1.0)
        q_new[3:7] /= np.linalg.norm(q_new[3:7])
        return q_new, cost0

    def _elastic(self, H, g, G, h, Gh, hh):
        nv, nc = self.nv, len(h)
        n = nv + nc
        P = np.zeros((n, n))
        P[:nv, :nv] = 2 * H
        qv = np.concatenate([2 * g, 1e4 * np.ones(nc)])
        rows = [np.hstack([G, -np.eye(nc)]), np.hstack([np.zeros((nc, nv)), -np.eye(nc)]),
                np.hstack([Gh, np.zeros((len(hh), nc))])]
        T = np.zeros((nv + 1, n))
        T[1:, :nv] = -np.eye(nv)
        A = sparse.csc_matrix(np.vstack(rows + [T]))
        b = np.concatenate([h, np.zeros(nc), hh, [TRUST_REGION], np.zeros(nv)])
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        try:
            sol = clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P)), qv, A, b,
                                         [clarabel.NonnegativeConeT(2 * nc + len(hh)),
                                          clarabel.SecondOrderConeT(nv + 1)], settings).solve()
            x = np.array(sol.x)
        except Exception:
            return np.zeros(nv)
        return x[:nv] if len(x) and np.all(np.isfinite(x)) else np.zeros(nv)

    # ------------------------------------------------- optional contact cost
    def _demo_contacts(self):
        """Per-frame limbs the demonstrator has on the object / terrain (from the demonstration only)."""
        inst = self.inst
        J = np.asarray(inst["human_joints"], dtype=float)
        names = inst["joint_names"]
        T = len(J)
        src = inst["source_scene"]
        out = {}
        if self.task == "object":
            P = np.asarray(src["boxes"][0]["poses"], dtype=float)
            step = np.concatenate([[0.0], np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1)])
            carried = (P[:, 2] - P[0, 2] > 0.03) | (step > 0.003)
            for side, jn in (("left_hand", "L_Wrist"), ("right_hand", "R_Wrist")):
                out[side] = carried & (np.linalg.norm(J[:, names.index(jn)] - P[:, :3], axis=1) <= 0.40)
        else:
            def sd(jn):
                p = J[:, names.index(jn)]
                d = np.full(T, np.inf)
                for b in src["boxes"]:
                    pb = np.asarray(b["pose"], dtype=float)
                    R = quat_to_mat(pb[3:7])
                    loc = (p - pb[:3]) @ R
                    qd = np.abs(loc) - np.asarray(b["half_size"])
                    d = np.minimum(d, np.linalg.norm(np.maximum(qd, 0), axis=1) + np.minimum(qd.max(1), 0))
                return d
            for side, pre in (("left", "Left"), ("right", "Right")):
                out[f"{side}_hand"] = np.minimum(sd(f"{pre}HandMiddle3"), sd(f"{pre}Hand")) <= 0.05
                out[f"{side}_foot"] = (sd(f"{pre}ToeBase") <= 0.06) | (sd(f"{pre}Foot") <= 0.11)
        return out

    def _add_contact_cost(self, t, H, g):
        """w * (sdf(p) - r)^2 for robot hand / sole points on demonstrated contacts, linearised."""
        if not hasattr(self, "_contacts"):
            self._contacts = self._demo_contacts()
            m = self.model
            self._limb_pts = {
                "left_hand": [(m.geom("left_sphere_hand_link").id, 0.04)],
                "right_hand": [(m.geom("right_sphere_hand_link").id, 0.04)],
                "left_foot": [(m.geom(f"left_ankle_roll_sphere_{k}_link").id, 0.005) for k in (1, 2, 3, 4)],
                "right_foot": [(m.geom(f"right_ankle_roll_sphere_{k}_link").id, 0.005) for k in (1, 2, 3, 4)],
            }
            kind = "object" if self.task == "object" else "terrain"
            self._contact_geoms = [m.geom(f"scene_{kind}_{k}").id for k, b in enumerate(self.scene["boxes"])
                                   if b.get("kind") == kind]
        for limb, want in self._contacts.items():
            if not want[t]:
                continue
            pts = self._limb_pts[limb]
            for gid, radius in pts:
                p = self.data.geom_xpos[gid].copy()
                best = min((self._scene_sdf_grad(gs, p) for gs in self._contact_geoms), key=lambda x: x[0])
                r = best[0] - radius
                a = best[1] @ self._point_jac(int(self.model.geom_bodyid[gid]), p)
                w = self.contact_weight / len(pts)
                H += w * np.outer(a, a)
                g += w * r * a

    def initial_q(self) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[:3] = self.src_kp[0, 0]
        q[3:7] = [np.cos(self.yaw0 / 2), 0, 0, np.sin(self.yaw0 / 2)]
        return q

    def run(self, q_nominal: np.ndarray | None = None) -> np.ndarray:
        q = q_nominal[0].copy() if q_nominal is not None else self.initial_q()
        init_feet = None
        if q_nominal is not None:
            self._fk(q_nominal[0], 0)
            init_feet = {s: [self.data.xpos[b].copy() for b in bs] for s, bs in self.foot_bodies.items()}
        out = []
        q_prev = q.copy()
        for t in range(self.T):
            if self.mocap:
                set_frame(self.data, self.scene, self.mocap, t)
            V_src, L = self._frame_mesh(t)
            foot_ref = {}
            if t > 0:
                self._fk(q_prev, t)
                for side in ("left", "right"):
                    if self.stance[side][t]:
                        foot_ref[side] = [self.data.xpos[b].copy() for b in self.foot_bodies[side]]
            q_nom = q_nominal[t] if q_nominal is not None else None
            w_nom = W_NOMINAL * np.exp(-t / TAU_NOMINAL) if q_nominal is not None else 0.0
            last = np.inf
            for _ in range(ITERS_FIRST if t == 0 else ITERS):
                q, cost = self._iteration(q, q_prev, t, V_src, L, foot_ref, q_nom, w_nom,
                                          init_feet if t == 0 else None)
                if abs(cost - last) <= 1e-3 * abs(cost) + 1e-7:
                    break
                last = cost
            out.append(q.copy())
            q_prev = q.copy()
        return np.array(out)


def retarget(inst: dict, objective: str = "mesh", contact_weight: float = 0.0) -> np.ndarray:
    kw = {"objective": objective, "contact_weight": contact_weight}
    if inst.get("augmentation") and inst["task"] == "object":
        # Sec. III-B robot-object: solve the nominal motion, then the augmented one anchored to it (Eq. 4/5).
        q_nom = Retargeter(inst, inst["nominal_scene"], **kw).run()
        return Retargeter(inst, inst["target_scene"], **kw).run(q_nominal=q_nom)
    # Nominal scenes, and robot-terrain augmentation (re-solved with ground points in the mesh).
    return Retargeter(inst, inst["target_scene"], **kw).run()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--objective", choices=["mesh", "keypoint"], default="mesh",
                    help="keypoint = ablation without the interaction mesh (same constraints and pipeline)")
    ap.add_argument("--contact-weight", type=float, default=0.0,
                    help="optional task cost (App. B-5) pulling hands/feet onto surfaces the demonstrator touches")
    args = ap.parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.inputs.glob("*.json")):
        out = args.outputs / path.name
        if out.exists():
            continue
        inst = json.loads(path.read_text())
        q = retarget(inst, args.objective, args.contact_weight)
        out.write_text(json.dumps({"q": np.round(q, 6).tolist()}))
        print(f"{inst['instance_id']}: {len(q)} frames", flush=True)


if __name__ == "__main__":
    main()
