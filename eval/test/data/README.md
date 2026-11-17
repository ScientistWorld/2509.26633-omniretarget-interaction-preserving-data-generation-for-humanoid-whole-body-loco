# Private corpus provenance

`test_demonstrations.npz` is generated data committed with the gym. It contains all 99 usable large-box clips of OMOMO subjects sub7/sub16, cropped from one second before object motion to at most 600 frames at 30 Hz, and the full downsampled climbing recordings 0/2/3 with their terrain box vertices. No method output or metric is stored in the pack. The fixed hidden constructor produces 325 instances, including two independent scene variations per object clip.

OMOMO source: the public InterMimic-processed `OMOMO_new.zip`, Google Drive id `141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j`. A private rebuild can fetch it with:

```bash
mkdir -p eval/test/local_tmp/omomo_full
curl -fL --retry 3 -o eval/test/local_tmp/omomo_full/OMOMO_new.zip \
 'https://drive.usercontent.google.com/download?id=141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j&export=download&confirm=t'
```

Climbing source: `amazon-far/holosoma`, pinned commit `c9cd0975be3e4769b3f91cecc2cb9e65c8eecabd`, under `src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/mocap_climb_seq_{0,2,3}`. The existing pack can supply these unchanged arrays with `--reuse-climb`; a complete source extraction instead accepts `--climb-dir` pointing to the fetched climbing directory.

Run the extraction only in a compute job:

```bash
python eval/test/build_test_data.py --omomo-zip eval/test/local_tmp/omomo_full/OMOMO_new.zip \
 --reuse-climb eval/test/data/test_demonstrations.npz
```

Fresh clones already contain the generated pack and need no private download or extraction. All private source artifacts, construction code, raw trajectories and per-recording diagnostics remain under `eval/test/`. Public download commands retain only the four train subjects and climbing sequence 1.
