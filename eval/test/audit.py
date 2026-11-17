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

ROOT = Path('/home/user')
REG = ROOT/'scripts/retargeters'
PRIVATE = ROOT/'eval/test'
# Resume only stages produced by the same scientific and evaluator source.
paths = sorted(p for d in ('method','baseline','data','eval/train','redteam') for p in (ROOT/d).rglob('*.py'))
paths += [PRIVATE/'scoring_core.py', PRIVATE/'hidden_tasks.py', Path(__file__), PRIVATE/'summarize.py']
fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()[:12]
OUT = PRIVATE/'audit_results'/fingerprint
OUT.mkdir(parents=True, exist_ok=True)
SHIPPED = {p.name:p.read_text() for p in REG.glob('*.sh') if p.stem in ('omniretarget','keypoint_ik')}
if len(SHIPPED)!=2:
    raise RuntimeError('restore the two shipped registrations before running audit')
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
    if (dest/'done.json').exists(): return
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
    (dest/'done.json').write_text(json.dumps({'fingerprint':fingerprint}))
    restore()

def m(flags=''):
    return 'python method/omniretarget.py --inputs "$1" --outputs "$2" '+flags

try:
    restore()
    subprocess.run([sys.executable,'-m','data.tasks','--out','data/downloads/tasks/train'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','method.check_geometry','data/downloads/tasks/train/inputs',
                    str(OUT/'collision_derivatives.json')],cwd=ROOT,check=True)
    stage('official', {Path(n).stem:s.split('cd /home/user\n')[-1] for n,s in SHIPPED.items()}, ('train','test'))
    honest = {
      'var_no_mesh':m('--objective keypoint'),
      'var_contact10':m('--contact-weight 10'),
      'var_contact100':m('--contact-weight 100'),
      'var_contact1000':m('--contact-weight 1000'),
      'var_keypoint_contact':m('--objective keypoint --contact-weight 100'),
      'var_private_search':'python -m method.audit_variants private_search "$1" "$2"',
      'var_support10':'python -m method.audit_variants contact_support10 "$1" "$2"',
      'var_support100':'python -m method.audit_variants contact_support100 "$1" "$2"',
      'var_small_steps':'python -m method.audit_variants contact_small_steps "$1" "$2"',
    }
    for n,body in honest.items(): stage(n,{n:body},('test','train'))
    stage('repeat',{'var_repeat':m()},('test',))
    cheats=['constant','impossible','nan','malformed','empty','partial','joint','velocity','quaternion','teleport','disconnected','mutable','filter','forgery','hidden_read']
    bodies={'sc_'+n:f'python redteam/shortcuts.py {n} "$1" "$2"' for n in cheats}
    bodies['fresh_unseen_candidate']='python redteam/shortcuts.py constant "$1" "$2"'
    bodies['sc_always_contact']='python -m method.audit_variants always_contact "$1" "$2"'
    stage('shortcuts',bodies,('test',))
    stage('constant_train',{'sc_constant':bodies['sc_constant']},('train',))
    # Merge audit-only results; official scoring/ files are restored below.
    for split in ('test','train'):
        scores={'experiments':{}}
        diags={'methods':{}}
        for dest in sorted(OUT.iterdir()):
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
