"""One sequential final-code verification job. Official results contain only shipped methods.

All held-out raw trajectories and diagnostics remain here. Each stage scores the same complete
split through the normal no-argument shell entrypoints. Checkpoints resume only this exact source
fingerprint; no overlapping training/inference processes or wall-clock method limits are used.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

def atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)

ROOT = Path('/home/user')
REG = ROOT/'scripts/retargeters'
PRIVATE = ROOT/'eval/test'
# The plan is independent of temporary on-disk registrations. Infrastructure may
# restart the job at any instruction, including after installing a candidate.
PLAN = json.loads((PRIVATE/'audit_plan.json').read_text())
SHIPPED = {n+'.sh':'#!/bin/bash\nset -e\ncd /home/user\n'+body+'\n'
           for n,body in PLAN['shipped'].items()}
# Cache identity covers scientific sources, assets, commands, dependencies and
# ranking definitions. Orchestration-only repairs do not invalidate completed runs.
paths = sorted(p for d in ('method','baseline','data','eval/train','redteam')
               for p in (ROOT/d).rglob('*.py'))
paths += [PRIVATE/'scoring_core.py', PRIVATE/'hidden_tasks.py', PRIVATE/'summarize.py',
          PRIVATE/'audit_plan.json', ROOT/'scripts/evaluate_train.sh',
          ROOT/'scripts/evaluate_test.sh', ROOT/'environment/setup.sh']
paths += sorted((PRIVATE/'assets').glob('*')) + sorted((PRIVATE/'data').glob('*.npz'))
paths += sorted((ROOT/'eval/train/assets').glob('*'))
manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
spec = json.loads((ROOT/'scoring/reference.json').read_text())['experiments']
manifest['_scoring_definitions'] = {e:{k:v for k,v in x.items()
                                        if k in ('weight','metrics','primary_metric')}
                                    for e,x in spec.items()}
manifest_text = json.dumps(manifest,sort_keys=True,separators=(',',':'))
science_digest = hashlib.sha256(manifest_text.encode()).hexdigest()
OUT = PRIVATE/'audit_results'/science_digest[:12]
for receipt in sorted((PRIVATE/'audit_results').glob('*/science_manifest.json')):
    if json.loads(receipt.read_text()) == manifest:
        OUT = receipt.parent
        break
OUT.mkdir(parents=True,exist_ok=True)
(PRIVATE/'audit_results').mkdir(exist_ok=True)
atomic_json(OUT/'science_manifest.json',manifest)
fingerprint = OUT.name
FILES = ['scores_train.json','scores_test.json','diagnostics_train.json','diagnostics_test.json']

def restore():
    for p in REG.glob('*.sh'): p.unlink()
    for n,s in SHIPPED.items(): (REG/n).write_text(s)
    for f in FILES:
        p=OUT/'official'/f
        if p.exists():
            if f == 'diagnostics_test.json':
                obj=json.loads(p.read_text())
                obj['methods']={n:{k:v for k,v in row.items() if k!='per_instance'} for n,row in obj['methods'].items()}
                (ROOT/'scoring'/f).write_text(json.dumps(obj,indent=2))
            else:
                shutil.copy2(p,ROOT/'scoring'/f)
    if (ROOT/'scoring/scores_train.json').exists():
        shutil.copy2(ROOT/'scoring/scores_train.json',ROOT/'scoring/scores.json')
        shutil.copy2(ROOT/'scoring/diagnostics_train.json',ROOT/'scoring/diagnostics.json')

def stage(name, methods, splits):
    dest=OUT/name
    if (dest/'done.json').exists():
        checkpoint=json.loads((dest/'done.json').read_text())
        if checkpoint.get('fingerprint') != fingerprint:
            raise RuntimeError(f'incompatible stage checkpoint: {dest}')
        print(name,'resumed completed stage',flush=True)
        return
    dest.mkdir(exist_ok=True)
    for p in REG.glob('*.sh'): p.unlink()
    for n,body in methods.items():
        (REG/(n+'.sh')).write_text('#!/bin/bash\nset -e\ncd /home/user\n'+body+'\n')
    env={**os.environ,'RETARGET_KEEP_OUTPUTS':str(dest/'outputs')}
    for split in splits:
        with (dest/(split+'.log')).open('w') as log:
            subprocess.run(['bash',f'scripts/evaluate_{split}.sh'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for f in (f'scores_{split}.json',f'diagnostics_{split}.json'):
            src=PRIVATE/'diagnostics_latest.json' if f=='diagnostics_test.json' else ROOT/'scoring'/f
            shutil.copy2(src,dest/f)
        print(name,split,'complete',flush=True)
    atomic_json(dest/'done.json',{'fingerprint':fingerprint,'science_digest':science_digest})
    restore()

try:
    restore()
    subprocess.run([sys.executable,'-m','data.tasks','--out','data/downloads/tasks/train'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','method.check_geometry','data/downloads/tasks/train/inputs',
                    str(OUT/'collision_derivatives.json')],cwd=ROOT,check=True)
    for item in PLAN['stages']:
        stage(item['name'],item['methods'],tuple(item['splits']))
    # Merge audit-only results; official scoring/ files are restored below.
    for split in ('test','train'):
        scores={'experiments':{}}
        diags={'methods':{}}
        for item in PLAN['stages']:
            dest=OUT/item['name']
            if not (dest/'done.json').exists():
                raise RuntimeError(f'incomplete audit stage: {dest}')
            sf=dest/f'scores_{split}.json'; df=dest/f'diagnostics_{split}.json'
            if not sf.exists() or not df.exists(): continue
            x=json.loads(sf.read_text())
            for e,v in x['experiments'].items():
                scores['experiments'].setdefault(e,{'results':{}})['results'].update(v['results'])
            diags['methods'].update(json.loads(df.read_text())['methods'])
        combined=OUT/('combined_'+split);combined.mkdir(exist_ok=True)
        (combined/f'scores_{split}.json').write_text(json.dumps(scores,indent=2))
        (combined/f'diagnostics_{split}.json').write_text(json.dumps(diags))
        subprocess.run([sys.executable,str(PRIVATE/'summarize.py'),str(combined),split],cwd=ROOT,check=True)
    # Populate scaled gym rows from final measured official results; preserve paper metadata.
    refp=ROOT/'scoring/reference.json'; ref=json.loads(refp.read_text())
    off=json.loads((OUT/'official/scores_test.json').read_text())['experiments']
    for e,x in ref['experiments'].items():
        x['results']={n:{'type':'proposed' if n=='omniretarget' else 'baseline',**row} for n,row in off[e]['results'].items()}
    refp.write_text(json.dumps(ref,indent=2)+'\n')
    (PRIVATE/'audit_results/latest.txt').write_text(fingerprint+'\n')
finally:
    restore()
subprocess.run([sys.executable,'validate.py'],cwd=ROOT,check=True)
print('FINAL AUDIT FINISHED',OUT,flush=True)
