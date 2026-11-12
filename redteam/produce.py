"""Actual candidate producers. Only visible inputs are used here.

All shortcuts are named as shortcuts. This module never reads eval/test,
scores_test, the paper, or reference metric values to produce an artifact.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import random
import runpy
import sys

import numpy as np
from scipy.optimize import minimize

from data.planar_robot import KEYPOINTS, NAME_TO_IDX, PlanarHumanoid, initial_guess

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
OUTPUTS = ROOT / 'checkpoints/retargeting'
TASKS = ['object_carry', 'terrain_step']


def load_task(name):
    return json.loads((ROOT / 'data/synthetic_tasks' / f'{name}.json').read_text())


def save(task, points, qs=None, method='omniretarget', **extra):
    if qs is None:
        qs = np.zeros((len(points), 11))
    artifact = dict(method=method, task=task['task'], dt=task['dt'],
                    q=np.asarray(qs).tolist(), keypoint_names=KEYPOINTS,
                    retargeted_keypoints=np.asarray(points).tolist(), **extra)
    (OUTPUTS / f'{method}_{task["task"]}.json').write_text(json.dumps(artifact, indent=2))


def reference_artifact(name):
    return json.loads((AUDIT / 'reference_seed0/artifacts' / f'omniretarget_{name}.json').read_text())


def height(x, task):
    return 0.32 * np.clip((np.asarray(x) - .4) / .15, 0, 1) if task == 'terrain_step' else np.zeros_like(x, dtype=float)


def signed_box(points, obj):
    diff = np.abs(points - np.array(obj['center'])) - np.array(obj['size']) / 2
    return np.linalg.norm(np.maximum(diff, 0), axis=-1) + np.minimum(np.max(diff, axis=-1), 0)


def module_from_source(source):
    namespace = dict(__name__='audit_candidate', __file__=str(ROOT / 'method/omniretarget.py'))
    exec(compile(source, str(ROOT / 'method/omniretarget.py'), 'exec'), namespace)
    return namespace


def run_reference(case):
    seed = int(case[-1])
    random.seed(seed)
    np.random.seed(seed)
    module = module_from_source((ROOT / 'method/omniretarget.py').read_text())
    for name in TASKS:
        module['run'](ROOT / f'data/synthetic_tasks/{name}.json', OUTPUTS / f'omniretarget_{name}.json')


def stronger_budget():
    # A real extra-compute intervention: unchanged objectives and constraints.
    source = (ROOT / 'method/omniretarget.py').read_text()
    source = source.replace('"maxiter": 140', '"maxiter": 700')
    source = source.replace('"maxiter": 220', '"maxiter": 1100')
    source = source.replace('"ftol": 1e-8', '"ftol": 1e-11')
    module = module_from_source(source)
    for name in TASKS:
        module['run'](ROOT / f'data/synthetic_tasks/{name}.json', OUTPUTS / f'omniretarget_{name}.json')


def weaker_no_mesh():
    # Remove the paper's interaction deformation term, retaining constraints.
    source = (ROOT / 'method/omniretarget.py').read_text()
    source = source.replace('return float(e_lap + e_key + e_smooth)', 'return float(e_key + e_smooth)')
    module = module_from_source(source)
    for name in TASKS:
        module['run'](ROOT / f'data/synthetic_tasks/{name}.json', OUTPUTS / f'omniretarget_{name}.json')


def better_contact_refinement():
    """Improve-mode constrained arm optimization with actual robot FK outputs.

    Keep the reference pelvis/legs/torso, so its collision and skating
    constraints are retained. Optimize the four arm angles to preserve object
    contacts, with joint limits, per-step arm limits, and all point collision
    inequalities. Terrain outputs remain the reference, already near perfect
    on hand-contact preservation. No test-slice metric is consulted.
    """
    robot = PlanarHumanoid()
    arm_indices = [7, 8, 9, 10]
    lower = np.array([-2.4, .1, -.2, -2.4])
    upper = np.array([.2, 2.4, 2.4, -.1])
    increments = np.array([.30, .36, .30, .36])
    report = []
    for name in TASKS:
        task = load_task(name)
        ref = reference_artifact(name)
        qs = np.asarray(ref['q'])
        if name == 'object_carry':
            for t, q_base in enumerate(qs.copy()):
                obj = task['objects'][t]
                source = np.asarray(task['source_keypoints'][t])
                prev = qs[t - 1, arm_indices] if t else None
                lo = np.maximum(lower, prev - increments) if t else lower
                hi = np.minimum(upper, prev + increments) if t else upper

                def points(a):
                    q = q_base.copy()
                    q[arm_indices] = a
                    return robot.fk(q)

                def objective(a):
                    pts = points(a)
                    hands = pts[[8, 11]]
                    contact = np.sum(signed_box(hands, obj) ** 2)
                    matching = np.sum((pts[[7, 8, 10, 11]] - source[[7, 8, 10, 11]]) ** 2)
                    return float(15 * contact + .03 * matching + (.003 * np.sum((a - prev) ** 2) if t else 0))

                def nonpenetration(a):
                    pts = points(a)
                    return np.r_[pts[:, 1], signed_box(pts[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10]], obj)]

                result = minimize(objective, np.clip(q_base[arm_indices], lo, hi), method='SLSQP',
                                  bounds=list(zip(lo, hi)), constraints=[{'type':'ineq', 'fun':nonpenetration}],
                                  options={'maxiter':250, 'ftol':1e-11})
                if np.min(nonpenetration(result.x)) >= -1e-6:
                    qs[t, arm_indices] = result.x
                report.append(dict(task=name, frame=t, success=bool(result.success),
                                   violation=float(max(0, -np.min(nonpenetration(result.x))))))
        save(task, np.array([robot.fk(q) for q in qs]), qs)
    (AUDIT / 'better_contact_refinement/optimizer.json').write_text(json.dumps(report, indent=2))


def weak_ik():
    from baseline.keypoint_ik import run
    for name in TASKS:
        path = OUTPUTS / f'omniretarget_{name}.json'
        run(ROOT / f'data/synthetic_tasks/{name}.json', path)
        out = json.loads(path.read_text())
        out['method'] = 'omniretarget'
        path.write_text(json.dumps(out, indent=2))


def fk_batch(q):
    """Vectorized exact FK for finite full-configuration grid enumeration."""
    x,z,torso,lh,lk,rh,rk,ls,le,rs,re = q.T
    pelvis = np.stack([x,z], axis=-1)
    def v(a, length, sign=-1):
        return length * np.stack([np.sin(a), sign * np.cos(a)], axis=-1)
    neck = pelvis + v(torso, .48, 1)
    lsh = neck + [-.16, -.06]; rsh = neck + [.16, -.06]
    lkn = pelvis + v(lh, .40); rkn = pelvis + v(rh, .40)
    lf = lkn + v(lh+lk, .39); rf = rkn + v(rh+rk, .39)
    lel = lsh + v(ls, .27); rel = rsh + v(rs, .27)
    return np.stack([pelvis,neck,lkn,lf,rkn,rf,lsh,lel,lel+v(ls+le,.25),rsh,rel,rel+v(rs+re,.25)], axis=1)


def brute_force_grid():
    # Enumerate every candidate in a 3^11 grid per frame. This is an honest
    # alternative, not a shortcut unless the resulting coarse grid wins.
    from method.omniretarget import stance_feet
    grid = np.array(list(itertools.product([-1., 0., 1.], repeat=11)))
    scales = np.array([.12,.15,.25,.45,.5,.45,.5,.5,.5,.5,.5])
    qmin = np.array([-2,.45,-.7,-1.4,-.2,-1.4,-1.9,-2.4,.1,-.2,-2.4])
    qmax = np.array([2,1.45,.7,1.4,1.9,1.4,.2,.2,2.4,2.4,-.1])
    counts = {}
    for name in TASKS:
        task = load_task(name); source = np.array(task['source_keypoints'])
        qs=[]; pts=[]
        for t, src in enumerate(source):
            candidates = np.clip(initial_guess(src) + grid * scales, qmin, qmax)
            p = fk_batch(candidates)
            loss = np.sum((p - src) ** 2, axis=(1,2))
            loss += 1000 * np.sum(np.maximum(0, height(p[:,:,0], name) - p[:,:,1]) ** 2, axis=1)
            obj = task['objects'][t]
            if obj:
                loss += 1000 * np.sum(np.minimum(0,signed_box(p[:,[0,1,2,3,4,5,6,7,9,10]],obj)) ** 2,axis=1)
                loss += 10 * np.sum(signed_box(p[:,[8,11]],obj) ** 2,axis=1)
            for hand in task['contacts'][t].get('terrain',[]):
                if 'hand' in hand:
                    idx=NAME_TO_IDX[hand]
                    loss += 10 * (p[:,idx,1]-height(p[:,idx,0],name))**2
            if t:
                for foot in stance_feet(source[t-1],src,task['dt'],name):
                    loss += 1000*np.sum((p[:,NAME_TO_IDX[foot]]-pts[-1][NAME_TO_IDX[foot]])**2,axis=1)
            best = int(np.argmin(loss)); qs.append(candidates[best]);pts.append(p[best])
        save(task,np.array(pts),np.array(qs))
        counts[name]=len(grid)*len(source)
    (AUDIT/'brute_force_grid/enumeration.json').write_text(json.dumps(counts,indent=2))


def shortcut_unchecked_keypoints():
    """Shortcut: independently place public source joints, bypass morphology."""
    for name in TASKS:
        task = load_task(name); src = np.array(task['source_keypoints']); pts=src.copy()
        n=len(src)
        # Connect stance edges for lags 1 and 2 on the complete public motion,
        # with the stance definition available in eval/train. No hidden index
        # or answer file is read.
        for foot in [3,5]:
            parent=list(range(n))
            def find(i):
                while parent[i] != i:
                    parent[i]=parent[parent[i]]; i=parent[i]
                return i
            for lag in [1,2]:
                for t in range(lag,n):
                    a=t-lag
                    v=abs(src[t,foot,0]-src[a,foot,0])/(lag*task['dt'])
                    h=height(src[t,foot,0],name);hp=height(src[a,foot,0],name)
                    if v<.35 and abs(h-hp)<.05 and src[t,foot,1]<=h+.03:
                        parent[find(t)]=find(a)
            for root in set(find(i) for i in range(n)):
                group=[i for i in range(n) if find(i)==root]
                p=src[group,foot].mean(axis=0);p[1]=max(p[1],height(p[0],name))
                pts[group,foot]=p
        for t in range(n):
            for i in range(12):
                pts[t,i,1]=max(pts[t,i,1],height(pts[t,i,0],name))
            obj=task['objects'][t]
            if obj:
                c=np.array(obj['center']);half=np.array(obj['size'])/2
                for i in range(12):
                    p=pts[t,i]; delta=p-c
                    if i in [8,11] or signed_box(p,obj)<0:
                        if np.all(np.abs(delta)<=half):
                            axis=int(np.argmin(half-np.abs(delta)))
                            p[axis]=c[axis]+(1 if delta[axis]>=0 else -1)*half[axis]
                        else:
                            p[:]=c+np.clip(delta,-half,half)
            for hand in task['contacts'][t].get('terrain',[]):
                if 'hand' in hand:
                    idx=NAME_TO_IDX[hand];pts[t,idx,1]=height(pts[t,idx,0],name)
        save(task,pts)  # q is deliberately unrelated to these keypoints.


def simple_output(case):
    robot=PlanarHumanoid()
    for name in TASKS:
        task=load_task(name);src=np.array(task['source_keypoints'])
        if case=='shortcut_copy_input' or case=='shortcut_score_forgery':
            pts=src
        elif case=='shortcut_constant':
            q=initial_guess(src[0]); q[1]+=1
            pts=np.repeat(robot.fk(q)[None],len(src),axis=0)
        elif case=='shortcut_nan':
            pts=np.full_like(src,np.nan)
        elif case=='shortcut_impossible_keypoints':
            pts=np.full_like(src,1e200)
        elif case=='shortcut_impossible_q':
            artifact=reference_artifact(name)
            pts=np.array(artifact['retargeted_keypoints'])
            save(task,pts,np.full((len(src),11),1e200),success=True,runtime=-1e200,
                 claimed_contact=1,claimed_penetration=-1)
            continue
        elif case=='shortcut_train_filter':
            pts=np.zeros_like(src)
        elif case=='shortcut_unlisted_method':
            save(task,np.zeros_like(src),method='shortcut_unlisted_method')
            continue
        elif case=='shortcut_mutable_task':
            # Shortcut: replace the evaluator's public "trusted" instance.
            # Same frame count; source/labels/objects become a trivial task.
            pts=np.zeros_like(src)
            task['source_keypoints']=pts.tolist()
            task['objects']=[None]*len(src)
            task['contacts']=[{'object':[],'terrain':['left_hand','right_hand']} for _ in src]
            (ROOT/f'data/synthetic_tasks/{name}.json').write_text(json.dumps(task))
        elif case=='shortcut_mutable_baseline':
            # Shortcut probe: worsen comparison artifacts while keeping the
            # candidate as the same physical but very weak constant robot.
            q=initial_guess(src[0]);q[1]+=1
            pts=np.repeat(robot.fk(q)[None],len(src),axis=0)
            save(task,np.full_like(src,-100),method='keypoint_ik')
        elif case=='shortcut_partial_missing':
            if name=='terrain_step':
                continue
            ref=reference_artifact(name)
            save(task,np.array(ref['retargeted_keypoints']),np.array(ref['q']))
            continue
        elif case=='shortcut_malformed':
            pts=np.zeros((0,12,2))
        else:
            raise ValueError(case)
        save(task,pts,success=True,time=0,metrics={'contact_preservation':100,'penetration_duration':-100})


def main():
    case=sys.argv[1]
    if case.startswith('reference_seed'):
        run_reference(case)
    elif case=='better_budget':
        stronger_budget()
    elif case=='better_contact_refinement':
        better_contact_refinement()
    elif case=='weaker_no_mesh':
        weaker_no_mesh()
    elif case=='weaker_keypoint_ik':
        weak_ik()
    elif case=='brute_force_grid':
        brute_force_grid()
    elif case=='shortcut_unchecked_keypoints':
        shortcut_unchecked_keypoints()
    elif case=='shortcut_empty':
        pass
    else:
        simple_output(case)


if __name__=='__main__':
    main()
