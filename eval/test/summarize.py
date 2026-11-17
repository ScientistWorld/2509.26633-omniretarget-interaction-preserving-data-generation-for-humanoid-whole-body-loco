"""Paired, recording-clustered scientific ranking audit; not a ranked score file."""
from pathlib import Path
import json
import re
import sys
import numpy as np
from eval.test.scoring_core import aggregate

root=Path('/home/user')
d=Path(sys.argv[1]);split=sys.argv[2]
spec=json.loads((root/'scoring/reference.json').read_text())['experiments']
scores=json.loads((d/f'scores_{split}.json').read_text())['experiments']
diag=json.loads((d/f'diagnostics_{split}.json').read_text())['methods']
names=sorted(diag)
vals={n:{e:scores[e]['results'][n] for e in spec} for n in names}
honest=[n for n in names if n in ('omniretarget','keypoint_ik','sc_constant') or n.startswith('var_')]

def scales(cohort):
    return {e:{m:float(np.std([vals[n][e][m] for n in cohort])) for m in x['metrics']} for e,x in spec.items()}

def rank(c,r,sd):
    return sum(x['weight']*abs(md['coefficient'])*(1 if md['higher_is_better'] else -1)*(c[e][m]-r[e][m])/sd[e][m]
               for e,x in spec.items() for m,md in x['metrics'].items() if sd[e][m]>1e-12)

sd=scales(names); hsd=scales(honest)
refper=diag['omniretarget'].get('per_instance')
out={'split':split,'normalization_cohort':names,'honest_cohort':honest,'sd':sd,'methods':{}}
if refper is None:
    raise RuntimeError('reference failed: inspect official diagnostics before claiming success')
# Resample complete recordings, shared between nominal and augmented windows. All methods get
# the same draws. Object clips and climbing sequences form separate strata.
groups={}
for iid in refper:
    mt=re.search(r'sub\d+_largebox_\d+|seq\d+',iid)
    g=mt.group() if mt else iid
    groups.setdefault(g,[]).append(iid)
strata=[[g for g in groups if g.startswith('sub')],[g for g in groups if g.startswith('seq')]]
out['paired_recording_clusters'] = {"object": len(strata[0]), "terrain": len(strata[1])}
out['instance_counts'] = {e: sum(v['experiment'] == e for v in refper.values()) for e in spec}
rng=np.random.default_rng(1701)
B=1000
picks=[]
for b in range(B):
    p=[]
    for gs in strata:
        if gs:
            for g in rng.choice(gs,len(gs),replace=True):p.extend(groups[g])
    picks.append(p)

def bv(per,b):
    ms={f'{k}':per[i] for k,i in enumerate(picks[b])}
    return aggregate(ms,{k:{'experiment':v['experiment']} for k,v in ms.items()})
rboot=[bv(refper,b) for b in range(B)]
for n in names:
    row={'status':diag[n]['status'],'reason':diag[n]['reason'],'cpu_s':diag[n].get('cpu_s'),
         'rank_delta':rank(vals[n],vals['omniretarget'],sd),
         'honest_cohort_rank_delta':rank(vals[n],vals['omniretarget'],hsd),'metrics':vals[n]}
    if diag[n]['status']=='ok' and n!='omniretarget':
        bs=[rank(bv(diag[n]['per_instance'],b),rboot[b],sd) for b in range(B)]
        row['paired_recording_bootstrap_ci95']=np.percentile(bs,[2.5,97.5]).tolist()
    if diag[n]['status']=='failed':
        assert all((vals[n][e][m]<=vals['omniretarget'][e][m] if md['higher_is_better'] else vals[n][e][m]>=vals['omniretarget'][e][m]) for e,x in spec.items() for m,md in x['metrics'].items())
    out['methods'][n]=row
out['reference_at_bound']={e:[m for m,md in x['metrics'].items() if vals['omniretarget'][e][m]==(1 if md['higher_is_better'] else 0)] for e,x in spec.items()}
out['constant_metrics']={e:[m for m in x['metrics'] if max(vals[n][e][m] for n in honest)==min(vals[n][e][m] for n in honest)] for e,x in spec.items()}
if 'var_repeat' in vals:
    out['repeat_max_abs_metric_difference']=max(abs(vals['omniretarget'][e][m]-vals['var_repeat'][e][m]) for e,x in spec.items() for m in x['metrics'])
    out['repeat_raw_equal']=diag['omniretarget']['per_instance']==diag['var_repeat'].get('per_instance')
(d/f'summary_{split}.json').write_text(json.dumps(out,indent=2)+'\n')
for n,v in sorted(out['methods'].items(),key=lambda z:-z[1]['rank_delta']):
    print(n,v['status'],round(v['rank_delta'],5),v.get('paired_recording_bootstrap_ci95'),v['reason'],flush=True)
print('bound',out['reference_at_bound'],'constant',out['constant_metrics'],flush=True)
