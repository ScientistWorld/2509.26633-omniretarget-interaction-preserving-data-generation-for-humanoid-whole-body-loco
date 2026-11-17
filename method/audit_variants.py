"""Honest audit variants; no evaluator imports or held-out data access."""
import argparse
import json
from pathlib import Path
import numpy as np
import method.omniretarget as M

p = argparse.ArgumentParser()
p.add_argument('variant')
p.add_argument('inputs', type=Path)
p.add_argument('outputs', type=Path)
a = p.parse_args()
if a.variant == 'private_search':
    M.ITERS_FIRST, M.ITERS = 150, 30
if a.variant == 'contact_small_steps':
    M.TRUST_REGION = 0.1
    M.ITERS_FIRST, M.ITERS = 100, 20

class Always(M.Retargeter):
    def _demo_contacts(self):
        return {k: np.ones_like(v) if k.endswith('hand') else v for k, v in super()._demo_contacts().items()}

class Support(M.Retargeter):
    """Task-cost variant: maintain demonstrated support as well as interaction contact.

    This changes the optimization objective, not the evaluator or its labels.
    Ground/terrain distance is linearized using the method's own scene geometry.
    """
    def _add_contact_cost(self, t, H, g):
        super()._add_contact_cost(t, H, g)
        if not hasattr(self, '_support_geoms'):
            self._support_geoms = [self.model.geom('ground').id] + [
                self.model.geom(f'scene_terrain_{k}').id
                for k, box in enumerate(self.scene['boxes']) if box.get('kind') == 'terrain']
            names = self.inst['joint_names']
            toes = [names.index(n) for n in M.TOES[self.fmt]]
            src = np.asarray(self.inst['human_joints'])[:, toes]
            height = src[:, :, 2] - M.support_height(src[:, :, :2], self.inst['source_scene'])
            self._supported = np.abs(height) < 0.15
        for k, side in enumerate(('left', 'right')):
            if not self._supported[t, k] or (t > 0 and not self.stance[side][t]):
                continue
            for gid, radius in self._limb_pts[side+'_foot']:
                point = self.data.geom_xpos[gid].copy()
                dist, normal = min((self._scene_sdf_grad(gs, point) for gs in self._support_geoms),
                                   key=lambda pair: pair[0])
                row = normal @ self._point_jac(int(self.model.geom_bodyid[gid]), point)
                weight = self.contact_weight / 4
                H += weight * np.outer(row, row)
                g += weight * (dist - radius) * row

def run_class(cls, inst, weight):
    kw = {'contact_weight': weight}
    if inst.get('augmentation') and inst['task'] == 'object':
        q0 = cls(inst, inst['nominal_scene'], **kw).run()
        return cls(inst, inst['target_scene'], **kw).run(q_nominal=q0)
    return cls(inst, inst['target_scene'], **kw).run()

a.outputs.mkdir(parents=True, exist_ok=True)
for f in sorted(a.inputs.glob('*.json')):
    inst = json.loads(f.read_text())
    if a.variant == 'always_contact':
        q = run_class(Always, inst, 100.0)
    elif a.variant.startswith('contact_support'):
        q = run_class(Support, inst, float(a.variant.removeprefix('contact_support')))
    elif a.variant == 'contact_small_steps':
        q = M.retarget(inst, contact_weight=100.0)
    else:
        q = M.retarget(inst)
    (a.outputs/f.name).write_text(json.dumps({'q': np.round(q, 6).tolist()}))
