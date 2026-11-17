"""Deliberate invalid/shortcut submissions for the final evaluator audit. No test answers."""
import json
import sys
from pathlib import Path
import numpy as np
from baseline.keypoint_ik import retarget

name, inputs, outputs = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
outputs.mkdir(parents=True, exist_ok=True)
if name == 'empty':
    raise SystemExit(0)
if name == 'hidden_read':
    # Must be denied by the kernel, including a chmod attempt, rather than just hidden by convention.
    secret = Path('/home/user/eval/test/data/test_demonstrations.npz')
    try:
        secret.read_bytes()
    except PermissionError:
        pass
    else:
        raise RuntimeError('hidden corpus was readable')
    try:
        secret.chmod(0o644)
    except PermissionError:
        pass
    try:
        secret.read_bytes()
    except PermissionError:
        pass
    else:
        raise RuntimeError('chmod bypassed the filesystem boundary')
for f in sorted(inputs.glob('*.json')):
    inst = json.loads(f.read_text())
    T = inst['num_frames']
    if name == 'partial' and (inst['task'] != 'object' or inst.get('augmentation')):
        continue
    J = np.asarray(inst['human_joints'])
    names = inst['joint_names']
    pel = J[0, names.index('Pelvis' if 'Pelvis' in names else 'Hips')] * inst['alpha']
    q = np.zeros((T, 36))
    q[:, :3] = [pel[0], pel[1], 0.793]
    q[:, 3] = 1
    extra = {}
    if name == 'impossible': q[:] = 1e200
    elif name == 'nan': q[:] = np.nan
    elif name == 'malformed':
        (outputs/f.name).write_text('{broken json')
        continue
    elif name == 'joint': q[T//2,10] = 4
    elif name == 'velocity': q[T//2,19] = 1.5
    elif name == 'quaternion': q[:,3] = 2
    elif name == 'teleport': q[T//2,0] += 1
    elif name == 'disconnected':
        extra = {'keypoints': (J * inst['alpha']).tolist(), 'valid': True, 'contact_preservation':1, 'penetration_duration':0}
    elif name == 'mutable':
        inst['human_joints'] = np.zeros_like(J).tolist()
        inst['target_scene']['boxes'] = []
        f.write_text(json.dumps(inst))
    elif name == 'filter':
        for sn in ('scores_train.json','scores_test.json'):
            try:
                Path('/home/user/scoring',sn).write_text('{}')
            except PermissionError:
                pass
    elif name == 'forgery':
        q = retarget(inst)
        extra = {'metrics': {'contact_preservation':1, 'penetration_duration':0}, 'valid':True, 'cpu_s':0}
        try:
            Path('/home/user/scoring/scores_test.json').write_text('{"experiments":{}}')
        except PermissionError:
            pass
    (outputs/f.name).write_text(json.dumps({'q': q.tolist(), **extra}))
