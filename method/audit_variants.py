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
class Always(M.Retargeter):
    def _demo_contacts(self):
        return {k: np.ones_like(v) if k.endswith('hand') else v for k, v in super()._demo_contacts().items()}
a.outputs.mkdir(parents=True, exist_ok=True)
for f in sorted(a.inputs.glob('*.json')):
    inst = json.loads(f.read_text())
    if a.variant == 'always_contact':
        kw = {'contact_weight': 100.0}
        if inst.get('augmentation') and inst['task'] == 'object':
            q0 = Always(inst, inst['nominal_scene'], **kw).run()
            q = Always(inst, inst['target_scene'], **kw).run(q_nominal=q0)
        else:
            q = Always(inst, inst['target_scene'], **kw).run()
    else:
        q = M.retarget(inst)
    (a.outputs/f.name).write_text(json.dumps({'q': np.round(q, 6).tolist()}))
