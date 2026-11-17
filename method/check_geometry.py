"""Numerically check the collision linearization on public training inputs (job only)."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from data.scene import set_frame
from method.omniretarget import Retargeter, COLLISION_THRESHOLD

p = argparse.ArgumentParser()
p.add_argument('inputs', type=Path)
p.add_argument('report', type=Path)
a = p.parse_args()
rng = np.random.default_rng(2401)
chosen = {}
for f in sorted(a.inputs.glob('*.json')):
    inst = json.loads(f.read_text())
    key = (inst['task'], bool(inst.get('augmentation')))
    chosen.setdefault(key, inst)
errors = []
for inst in chosen.values():
    r = Retargeter(inst, inst['target_scene'])
    for t in (0, r.T//2, r.T-1):
        if r.mocap:
            set_frame(r.data, r.scene, r.mocap, t)
        q = r.initial_q()
        # Check both deeply penetrating initialization and shallower configurations.
        for lift in (0.0, 0.2, 0.5):
            q[2] = r.initial_q()[2] + lift
            r._fk(q, t)
            phi, jac = r._collisions()
            pairs = sorted(set((int(r.data.contact[k].geom1), int(r.data.contact[k].geom2))
                               for k in range(r.data.ncon)))
            used = []
            for g1, g2 in pairs:
                gr, gs = (g1, g2) if r.model.geom_contype[g1] == 1 else (g2, g1)
                dist = mujoco.mj_geomDistance(r.model, r.data, gr, gs, COLLISION_THRESHOLD, None)
                if dist <= COLLISION_THRESHOLD:
                    used.append((gr, gs))
            assert len(used) == len(phi)
            for _ in range(4):
                direction = rng.normal(size=r.nv)
                direction /= np.linalg.norm(direction)
                ds = []
                eps = 1e-6
                for sign in (-1, 1):
                    perturbed = q.copy()
                    mujoco.mj_integratePos(r.model, perturbed, sign*eps*direction, 1.0)
                    r._fk(perturbed, t)
                    ds.append(np.array([mujoco.mj_geomDistance(r.model, r.data, gr, gs,
                                                             COLLISION_THRESHOLD, None)
                                        for gr, gs in used]))
                numerical = (ds[1]-ds[0])/(2*eps)
                # Threshold clipping and the zero-distance contact cusp are nonsmooth.
                smooth = (np.abs(phi) > 1e-4) & (phi < COLLISION_THRESHOLD-1e-4)
                errors.extend(np.abs((jac @ direction-numerical)[smooth]).tolist())
            r._fk(q, t)
if not errors:
    raise RuntimeError('no active distances checked')
report = {'count':len(errors), 'median_abs_error':float(np.median(errors)),
          'p90_abs_error':float(np.percentile(errors,90)), 'max_abs_error':max(errors)}
a.report.parent.mkdir(parents=True, exist_ok=True)
a.report.write_text(json.dumps(report,indent=2)+'\n')
print('Collision derivative check:',report,flush=True)
if report['median_abs_error'] > 1e-3 or report['p90_abs_error'] > 0.02:
    raise RuntimeError('collision distance derivative disagrees with finite differences')
