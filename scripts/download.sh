#!/bin/bash
# Fetch every external asset into data/downloads/ (idempotent; needs internet) and build the
# public train instances.  Estimated download: ~20 MB (holosoma files) + 0.9 GB (OMOMO zip,
# deleted after extraction; ~0.4 GB of train clips are kept).
#
# 1. holosoma_retargeting -- the paper authors' code release (Apache-2.0), pinned commit:
#    Unitree G1 29-DoF sphere-hand model + meshes, the OMOMO large-box mesh, the subject height
#    table and the public climbing MoCap sequence (seq 1) with its box terrain.
# 2. OMOMO box carrying, InterMimic-processed release (the format the paper code reads):
#    only the large-box clips of the public train subjects (sub3, sub8, sub10, sub12) are kept.
#    The held-out test subjects and climbing sequences live only in eval/test/.
# 3. data/tasks.py builds data/downloads/tasks/train/inputs/*.json.
# The robot model used by the gym (data/robot/g1_29dof_spherehand_hull.xml) is committed; it can
# be rebuilt from the fetched files with: python data/robot/build_g1_model.py
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
D=data/downloads
mkdir -p $D/holosoma $D/omomo/largebox

SHA=c9cd0975be3e4769b3f91cecc2cb9e65c8eecabd
BASE=https://raw.githubusercontent.com/amazon-far/holosoma/$SHA/src/holosoma_retargeting/holosoma_retargeting
while read -r f; do
  [ -z "$f" ] && continue
  if [ ! -s "$D/holosoma/$f" ]; then
    mkdir -p "$(dirname "$D/holosoma/$f")"
    curl -sfL --retry 3 -o "$D/holosoma/$f.part" "$BASE/$f"
    mv "$D/holosoma/$f.part" "$D/holosoma/$f"
  fi
done < data/holosoma_files.txt
# the model file references ../assets/half_sphere.obj relative to meshes/; keep a copy next to it
[ -s $D/holosoma/models/g1/half_sphere.obj ] || cp $D/holosoma/models/g1/assets/half_sphere.obj $D/holosoma/models/g1/half_sphere.obj

if [ ! -f "$D/omomo/largebox/.complete" ]; then
  ZIP=$D/omomo/OMOMO_new.zip
  if [ ! -s "$ZIP" ]; then
    curl -sfL --retry 3 -o "$ZIP.part" \
      "https://drive.usercontent.google.com/download?id=141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j&export=download&confirm=t"
    mv "$ZIP.part" "$ZIP"
  fi
  python - "$ZIP" $D/omomo/largebox <<'PY'
import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
n = 0
for name in z.namelist():
    b = os.path.basename(name)
    if "_largebox_" in b and b.split("_")[0] in ("sub3", "sub8", "sub10", "sub12"):
        open(os.path.join(sys.argv[2], b), "wb").write(z.read(name))
        n += 1
print(f"extracted {n} OMOMO train clips")
PY
  touch "$D/omomo/largebox/.complete"
  rm -f "$ZIP"
fi

if [ -z "$(ls -A $D/tasks/train/inputs 2>/dev/null)" ]; then
  python -m data.tasks --out $D/tasks/train
fi
