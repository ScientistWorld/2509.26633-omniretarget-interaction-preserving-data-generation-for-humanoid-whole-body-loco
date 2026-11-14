# Hardening report (iteration 12)

**Short answer: yes, better results score higher, and the review's main complaint is fixed.**

On the 180-instance hidden test (60 per experiment):
- A better method beats the reference beyond noise: **+0.15** [+0.09, +0.23].
- The no-interaction-mesh ablation now falls clearly below the reference: **−0.57** [−0.73, −0.41]. In iteration 11 it tied at −0.14.
- The keypoint-IK baseline is far below: −8.31.
- Every shortcut that does not produce a valid robot motion fails with the worst value in every metric.

**What this iteration did not achieve:** the review asked that keypoint matching with contact costs fall *below* the reference. It went from +1.66 to a tie (+0.05 [−0.04, +0.15]). With ten times heavier contact weights it beats the reference by +0.17, as much as the mesh with the same costs. That is a legitimate second route to better contact, and I report it as such; see "Could not fix".

### Where the numbers come from

- **Hidden-test runs:**
  - The shipped methods ran alone in the official run: `redteam/hardening/results/official/`, identical to `scoring/scores_*.json`.
  - Variants and shortcuts ran in resumable chunks through the unchanged `bash scripts/evaluate_test.sh` and `evaluate_train.sh`: `redteam/hardening/results/chunks/`.
  - Merged results and summaries: `redteam/hardening/results/combined_{test,train}/`.
- **Rank delta:** weight × |coefficient| × directed improvement / sd, summed over every metric and experiment.
  - **Normalizer ("cohort6"):** the population sd across six honest methods (reference, keypoint IK, no-mesh, contact-aware, keypoint+contact, constant pose), as in iteration 11.
  - **Confidence intervals:** paired bootstrap over test instances (`redteam/hardening/summarize.py`).
  - **Second normalizer ("nearref"):** the reference plus its close variants only. It is also reported, but its intervals are about ±2.5, because several metrics barely vary among near-reference methods.
- **Before vs after:** the "before" numbers are iteration 11's, on its own task and normalizer. Compare them by sign and order, not by size.
- **Noise:** the methods are deterministic. The reference rerun inside the tamper chunk is bit-identical to the official run, so seed noise is 0. The intervals capture which instances are sampled.

## What the review listed, and what changed

**1. "Without the interaction mesh the method ties the reference."**

Two things caused the tie.
- *Scene scale.* The robot acted in the *human-scale* scene. The box and the column sat at the demonstrator's height, while the robot was 0.87–1.11 times smaller or larger. The object-frame mesh then pulled the whole body toward them and the feet hovered: foot contact was 0.51–0.83.
- *Missing paper pieces.* The paper's augmentation formulation (Eq. 4/5) was absent.

Fixes, all from the paper:
- **Robot-scale scenes.**
  - *Paper basis:* Sec. III-A puts "the same manipulated object and environment points" in the source and target meshes, and App. A uses direct scaling.
  - *Change:* each instance now has a `nominal_scene`, the demonstration scene scaled by α = h_robot/h_human. The box keeps its true size, which reproduces the paper's "scaled hands end up inside the object" artifact for keypoint matching. The target scene is the nominal scene, or a modified copy of it for augmented instances.
  - *Fidelity:* both fidelity metrics now compare against the demonstration at the robot's scale, and the pelvis height is measured above the ground below it. The old metrics penalised every method by about 1.8 cm for walking the correctly scaled path, and penalised following a raised platform.
- **Object and terrain together** (the review's "object and terrain interactions"; paper Fig. 3): the box is now carried up a ramp onto a platform.
- **Eq. 4/5 restored for object augmentation.** The method first solves the nominal motion, then the augmented one with ‖q − q̄*‖²_W (W heavy on base and legs) and the initial feet fixed to the nominal ones.
- **Sec. III-B's "additional constraints" for terrain augmentation.** A foot the demonstrator plants is constrained onto the modified terrain in the frame it lands. The paper does not spell these constraints out; this is my reading, needed for "stable ground contact when the terrain is elevated".
  - *Before (dev split, terrain augmentation):* the reference's feet hovered over the raised or lowered platform, with foot contact 0.51–0.89 per instance.
- **Frame 0** (unspecified in the paper): the initially planted feet start on the ground.

The exploration job (`redteam/hardening/results/explore/`) compared candidate setups on a 60-instance dev split before this choice. These runs predate the landing constraints; the table shows rank deltas vs the reference.

| setup | reference contact (object/terrain/aug) | no-mesh | keypoint+contact | contact-aware |
|---|---|---:|---:|---:|
| C0: human-scale scenes (iteration 11) | 0.84 / 0.86 / 0.72 | +0.18 | +1.65 | +1.48 |
| C2: robot-scale scenes, larger augmentation | 0.99 / 0.98 / 0.87 | −0.54 | +0.28 | +0.38 |
| C3: C2 with ±12 % segment proportions | 0.97 / 0.90 / 0.83 | −0.41 | +0.89 | +0.95 |
| C4: C2 with the box carried up a ramp (**chosen**) | 0.99 / 0.98 / 0.86 | −0.32 | +0.23 | +0.35 |

**2. "A keypoint baseline with contact handling also beats the reference."**

It now ties instead of winning by +1.66. Its remaining edge is planted-foot contact in the nominal tasks: 1.0 against the reference's 0.98/0.99. The paper's per-frame formulation only sticks stance feet (3e). It does not force a landing foot onto the ground outside terrain augmentation, so I did not add that to the reference.

I also added `spurious_contact_duration`, the counterpart of contact preservation: a hand or foot touching a surface where the demonstrator had clearance, such as dragged swing feet. In the dev exploration, keypoint objectives dragged the swing foot along a raised platform in 23 % of swing frames, against 6 % for the mesh. In the final pipeline the differences are small (contributions ≤ 0.007), so the tie is not an artifact of this metric; it comes from the reference fixes above.

**3. "The keypoint baseline hit the 2400 s limit on a preempted node."**

- *Cause.* The job gets one CPU while the affinity mask shows eight. Multi-threaded BLAS then spin-waits: keypoint IK burned 144–208 CPU-s per instance, against 0.35 single-threaded.
- *Fix.* There is no wall-clock limit anywhere. The evaluator counts CPU time (user + system over the method's whole process tree, read from `/proc` while it runs, plus `RUSAGE_CHILDREN`) against **6 CPU-s per instance**. That is 1080 CPU-s for the test; the reference uses 127 (0.70 per instance). Methods run with one BLAS/OpenMP thread.
- *Hang guard.* A method whose process tree makes no CPU progress across 1800 consecutive one-second polls is treated as hung. Counting polls rather than seconds keeps a suspended node from triggering it.
- *Checks* (`results/budget_checks.json`, 5 CPU-s budget):

  | Test program | Outcome |
  |---|---|
  | Busy loop | fails |
  | Busy loop in a child process | fails |
  | `sleep` forever | fails as hung |
  | Normal program | passes |

## Do better results score higher? Before → after (rank delta vs reference, hidden test)

| variant | before (iter 11) | after | after, train | what it shows |
|---|---:|---:|---:|---|
| Contact-aware: reference + demonstration-derived hand/foot contact costs (App. B-5 task costs) | +1.472 | **+0.153** [+0.088, +0.229] | +0.180 | Genuine improvement. Contact reaches 1.0 in all three experiments, augmented penetration drops 0.017 → 0, skating stays ≤ 0.002. |
| Keypoint+contact, weight 100: same costs, 10× heavier | — | **+0.170** [+0.104, +0.249] | +0.174 | Honest alternative, as good as contact-aware; it is headroom. |
| Keypoint+contact: keypoint matching + contact costs + the reference's constraints and pipeline | +1.659 | **+0.054** [−0.036, +0.145] | +0.119 | Tie (see item 2). |
| Same, iteration-11 version: no frame-0 grounding, Eq. 4/5 or landing constraints | +1.659 | +0.002 [−0.071, +0.081] | +0.060 | Tie. |
| No mesh: keypoint objective, same constraints and pipeline | −0.138 | **−0.566** [−0.729, −0.413] | −0.429 | The mesh matters. Hand contact in augmentation 0.48 vs 0.92; augmented penetration 0.20 vs 0.017. |
| Keypoint-IK baseline (GMR/PHC family) | −7.385 | **−8.311** [−9.116, −7.558] | −8.100 | Paper direction (Table II): penetration 0.47/0.10/0.58 and skating 0.15/0.17/0.15, vs ≈0 for the reference. |
| Constant standing pose | −3.336 | −6.255 | −6.177 | Valid motion, scored honestly: contact 0.41/0.81/0.61, spurious contact 1.0/0.20/0.60, pelvis error 23–29 cm. |

Train and test rank the methods in the same order, and the reference scores alike on both: contact 0.984/0.989/0.955 on train vs 0.993/0.989/0.967 on test.

## Shortcuts: fix and score before → after

All shortcuts were run through `bash scripts/evaluate_test.sh` (no arguments), registered as methods.

| shortcut | before (iter 11 → its fix) | after (this gym) |
|---|---|---:|
| Disconnected body / unchecked keypoints (claimed keypoints over another `q`) | +3.12 → scored as its `q` | scored as its `q` (keypoint IK): **−8.311** |
| Score forgery (prewritten score files, claimed metrics) | −2.79 → scored as its `q` | **−8.311**; forged files overwritten |
| Mutable task: rewrite public train tasks and empty train scores during the test | +3.16 → no effect | No effect: the shipped methods' test scores are **bit-identical** to the official run (`combined_test/tamper_check.json`) |
| Impossible `q` (1e200), zero `q`, NaN, malformed (0 frames), empty | ties / undefined → failed | failed: **−120.7** (worst value in every metric) |
| Partial output / omit the failed experiment | 0 / +1.11 → failed | failed: **−120.7** |
| Joint-limit or velocity-limit violation | failed | failed: **−120.7** |
| Unlisted method name, train-filter | ties → all registered methods scored | every new `var_*`/`sc_*` name was discovered and scored |
| Wall-clock exhaustion / thread oversubscription (new) | — | counted CPU budget, single-threaded BLAS; busy and hung methods fail |
| Private search (new, `sc_select_best`): per instance, run four honest retargeters and keep the one that scores best under a public copy of the metrics, with labels inferred from the inputs (2.5 CPU-s/instance) | — | **+0.225** [+0.161, +0.297]; left open (below) |

## Shortcuts left open, and why

- **Private search over honest candidates.** `sc_select_best` scores +0.225. Its best single component scores +0.170, so the selection itself adds about +0.055, with overlapping intervals. Every component is an honest retargeter, and the selection criterion is the metrics themselves (penetration, skating, contact, fidelity). Those are the problem's objectives, and they must be computable from the inputs, because any method has to infer contacts and stance from the demonstration.
  - *Bound applied:* I cut the budget from 20 to 6 CPU-s per instance. That still leaves about 8× the reference, for hardware variation and heavier honest methods such as trajectory-wide optimisation, but caps such searches at about 8 candidates.
  - *Why not closed:* stopping a 4-candidate search would need about 2 CPU-s per instance, under 3× the reference. That would make the reference's own validity depend on the machine.
- **A method could read `eval/test/` while it runs.** That directory is absent from a future agent's workspace, so nothing can be precomputed offline. At test time, though, `hidden_tasks.py` is on disk.
  - *What it would yield:* only the annotations — stance frames, contact frames and α. Each is a deterministic function of the inputs the method already receives (planted feet, hands on surfaces, bone lengths).
  - *Mitigation:* `scoring/CONSTRAINTS.md` forbids reading `eval/`, and the separate grader checks for it.
- **Copying the better variants.** `redteam/hardening/variants.py` is visible. The separate grader judges novelty.

## Reference score before → after

The task changed: robot-scale scenes, the ramp, a new metric, and fidelity measured at robot scale. The two blocks are different problems.

| hidden test | penetration dur. / max (cm) | skating dur. / max (cm/s) | contact | spurious | pelvis (cm) | velocity (cm/s) |
|---|---|---|---|---|---|---|
| **before**, object | 0.006 / 0.07 | 0 / 0.37 | 0.827 | — | 1.04 | 4.45 |
| **before**, terrain | 0.013 / 0.20 | 0.002 / 0.72 | 0.856 | — | 0.99 | 9.07 |
| **before**, augmented | 0.005 / 0.16 | 0.001 / 0.67 | 0.714 | — | 1.45 | 7.20 |
| **after**, object | 0 / 0.004 | 0.0013 / 0.52 | 0.993 | 0.0005 | 0.71 | 2.99 |
| **after**, terrain | 0 / 0 | 0.0016 / 0.56 | 0.989 | 0.0004 | 0.69 | 6.78 |
| **after**, augmented | 0.017 / 0.19 | 0.0010 / 0.57 | 0.967 | 0.0095 | 1.98 | 7.10 |

- The official run of the shipped methods took 127 CPU-s for the reference and 62 for keypoint IK, on 180 instances.
- The paper's numbers are unchanged in `scoring/reference.json`. The planar setup is not on their cm scale.
- The augmented experiment's "proposed" row is this gym's own measured value, labelled as such. The paper reports no kinematic numbers for augmented motions, and validate.py requires a row.

## Headroom check

- **At or near the bound:** the reference sits close to it on penetration duration (0 / 0 / 0.017), skating duration (≤ 0.0016) and spurious contact (≤ 0.0095). Nominal contact is close to it (0.993 / 0.989).
- **Not at the bound:**
  - augmented contact, 0.967, mostly hands on a displaced or resized box (0.919);
  - augmented penetration, 0.017;
  - fidelity: pelvis 0.7 / 0.7 / 2.0 cm and velocity 3.0 / 6.8 / 7.1 cm/s. Keypoint IK reaches 0.2–1.3 cm and 0.6–1.9 cm/s, but fails every constraint, so a constrained method can still gain here.
- **What I added:** the augmentation experiment is larger, using paper-like ranges (box scaled 0.75–1.35 and displaced 6–16 cm; platform height × 0.4–1.8). This is where the paper's method falls short. The hidden test is 1.5× larger.
- **Test 1, a better method beats the reference:** contact-aware +0.15, heavy keypoint+contact +0.17, private search +0.22, all with intervals above zero. None reaches every bound: their fidelity errors stay at 0.5–1.9 cm and 2.8–7.7 cm/s.
- **Test 2, simple methods stay below:** keypoint IK −8.31, constant pose −6.26, no mesh −0.57. There is no brute force to scale against: the problem is continuous, and the paper's method does not exist to avoid enumeration.
- **Limit of this check:** under the cohort6 normalizer, the fidelity metrics count for little. The constant pose's 23–29 cm pelvis errors dominate their sd, so a future method's fidelity gains will show only if the ranked cohort is closer to the reference.

## Weights and coefficients

- Experiment weights are unchanged: 0.35 / 0.35 / 0.30.
- The new `spurious_contact_duration` has coefficient 0.8, the same as skating duration. Both are durations of a contact artifact, and the paper treats both kinds of artifact as critical for downstream RL.
- All other coefficients are unchanged: penetration 1.0 / 0.5, skating 0.8 / 0.4, contact 1.0, pelvis 0.3, velocity 0.3.

## Could not fix / stated plainly

- **Keypoint+contact ties the reference instead of falling below it.** That is +0.05 [−0.04, +0.15] on test and +0.12 [+0.01, +0.23] on train. A heavier-weighted version is above, at +0.17.
  - On these kinematic metrics, explicit contact objectives with the reference's hard constraints are as good as the mesh. The mesh's own advantage shows against keypoint matching *without* contact objectives (−0.57), and at equal contact weight (contact-aware +0.15 vs keypoint+contact +0.05).
  - The paper never runs a keypoint+contact+hard-constraint ablation. Its VideoMimic baseline uses soft costs.
  - Making this variant lose would mean weakening the metrics or turning the reference into more than the paper's method.
- **Keypoint IK slightly beats the reference on nominal terrain contact** (0.9994 vs 0.989). In robot-scale scenes the scaled keypoints already land on the surfaces. The paper's direction still holds on penetration, skating, object and augmented contact, and overall (−8.3).
- **The terrain-augmentation landing constraint is my reading** of Sec. III-B's unspecified "additional constraints". The frame-0 grounding is an initialisation choice the paper leaves open.
- **Budget change after the official run.** The official run used a 20 CPU-s/instance budget; it was tightened to 6 afterwards. The budget is only a pass/fail gate, and every reference, baseline and variant run used at most 0.76 CPU-s per instance (private search 2.5). The stored scores are therefore what the final code produces; they were not rerun.
- **Scaled substitute.** A planar 11-DoF robot with sampled link collisions and procedural demonstrations. There are no OMOMO/LAFAN1/MoCap data, no 3D G1/H1/T1 robots, a single GMR/PHC-family baseline, and no downstream RL or hardware.
- **Different subproblem solver:** SciPy SLSQP with an elastic collision slack instead of Drake's SOCP.
- **Infrastructure:** the compute node preempted the final job three times. The audit was made resumable in chunks; each chunk is recorded with its exit code.
- **validate.py warns** that `eval/test/hidden_tasks.py` regenerates data. This is by design: it builds the hidden test inputs; scoring reads only the methods' outputs.
