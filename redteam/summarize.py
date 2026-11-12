"""Analyze real test files after all producers and evaluators have exited."""
from __future__ import annotations

import json
from pathlib import Path
import math

import numpy as np
from data.planar_robot import PlanarHumanoid
from data.synthetic_locomotion import generate_task

ROOT=Path('/home/user')
AUDIT=ROOT/'redteam'
spec=json.loads((ROOT/'scoring/reference.json').read_text())['experiments']
ref=json.loads((AUDIT/'reference_seed0/scores_test.json').read_text())['experiments']


def read(case):
    path=AUDIT/case/'scores_test.json'
    return json.loads(path.read_text())['experiments'] if path.exists() else {}


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def scalar(exps,exp,metric):
    return exps.get(exp,{}).get('results',{}).get('omniretarget',{}).get(metric)


cases=[p.parent.name for p in AUDIT.glob('*/status.json')]
honest=[s for s in cases if not s.startswith('shortcut_')]
scales={}
for exp,espec in spec.items():
    scales[exp]={}
    for metric in espec['metrics']:
        values=[scalar(read(case),exp,metric) for case in honest]
        values=[v for v in values if finite(v)]
        stdev=float(np.std(values)) if values else 0
        scales[exp][metric]=stdev if stdev>1e-12 else 1.0

summary={'normalization_cohort':honest,'normalizers':scales,'cases':{},'seed_spread':{},
         'note':'Ranks are weighted differences from rerun reference, using honest-audit-cohort standard deviations; actual future cohort is unknown. Exact ties/Pareto claims do not need these scales.'}
for case in sorted(cases):
    status=json.loads((AUDIT/case/'status.json').read_text())
    exps=read(case);rank=0;components={};missing=[];nonfinite=[];dominates=True;strict=False
    for exp,espec in spec.items():
        components[exp]={}
        for metric,mspec in espec['metrics'].items():
            v=scalar(exps,exp,metric);rv=scalar(ref,exp,metric)
            if v is None:
                missing.append(f'{exp}.{metric}')
                dominates=False
                continue
            if not finite(v):
                nonfinite.append(f'{exp}.{metric}')
                dominates=False
                continue
            sign=1 if mspec['higher_is_better'] else -1
            gain=sign*(v-rv)
            if gain < -1e-9:dominates=False
            if gain > 1e-9:strict=True
            contribution=espec['weight']*abs(mspec['coefficient'])*gain/scales[exp][metric]
            rank+=contribution
            components[exp][metric]=contribution
    record=dict(status,rank_delta=rank if not nonfinite and status.get('test_exit')==0 else None,
                missing=missing,nonfinite=nonfinite,pareto_dominates=dominates and strict,
                components=components)
    physical={}
    for path in (AUDIT/case/'artifacts').glob('*.json'):
        out=json.loads(path.read_text())
        pts=np.array(out['retargeted_keypoints']);qs=np.array(out['q'])
        if len(qs)==len(pts) and qs.shape[1:]==(11,) and np.isfinite(qs).all() and np.isfinite(pts).all():
            fk=np.array([PlanarHumanoid().fk(q) for q in qs])
            # Avoid squaring absurd values in the explicit impossible-output probe.
            mismatch=float(np.max(np.abs(fk-pts))) if len(qs) else None
            lower=np.array([-2.,.45,-.7,-1.4,-.2,-1.4,-1.9,-2.4,.1,-.2,-2.4])
            upper=np.array([2.,1.45,.7,1.4,1.9,1.4,.2,.2,2.4,2.4,-.1])
            violation=float(max(0,np.max(lower-qs),np.max(qs-upper))) if len(qs) else None
            increments=np.array([.09,.08,.18,.24,.30,.24,.30,.30,.36,.30,.36])
            if out['task']=='terrain_step':
                increments=np.array([.16,.18,.24,.42,.55,.42,.55,.50,.60,.50,.60])
            velocity_violation=float(max(0,np.max(np.abs(np.diff(qs,axis=0))-increments))) if len(qs)>1 else None
            physical[out['task']]=dict(max_fk_coordinate_error_m=mismatch,
                                     max_joint_bound_violation=violation,
                                     max_increment_bound_violation=velocity_violation)
        else:
            physical[out['task']]={'invalid_or_nonfinite_artifact':True}
    record['independent_artifact_checks']=physical
    summary['cases'][case]=record

for exp,espec in spec.items():
    summary['seed_spread'][exp]={}
    for metric in espec['metrics']:
        values=[scalar(read(f'reference_seed{s}'),exp,metric) for s in range(3)]
        summary['seed_spread'][exp][metric]=dict(values=values,spread=max(values)-min(values))
summary['seed_rank_spread']=max(summary['cases'][f'reference_seed{s}']['rank_delta'] for s in range(3))-min(summary['cases'][f'reference_seed{s}']['rank_delta'] for s in range(3))
summary['data_generator_check']={}
for task in ['object_carry','terrain_step']:
    tracked=json.loads((ROOT/f'data/synthetic_tasks/{task}.json').read_text())
    regenerated=[generate_task(task,frames=len(tracked['source_keypoints']),seed=s) for s in range(3)]
    summary['data_generator_check'][task]=dict(seeds_0_1_2_identical=regenerated[0]==regenerated[1]==regenerated[2],
                                            identical_to_tracked=regenerated[0]==tracked)
(AUDIT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
for case,record in summary['cases'].items():
    print(case,'rank_delta=',record['rank_delta'],'test_exit=',record.get('test_exit'),
          'missing=',len(record['missing']),'nonfinite=',len(record['nonfinite']))
