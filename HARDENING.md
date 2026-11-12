# Hardening report (iteration 11)

**Short answer: yes, better results now score higher.** On a 120-instance hidden test, a better method scores clearly above the reference:
- A genuinely better variant beats the reference by **+1.47** (paired 95% CI [+1.07, +1.90]).
- The paper's keypoint-IK baseline family is far below it at **−7.39** [−8.18, −6.57].

Every shortcut from `REDTEAM.md` now either fails with the worst value in every metric or scores exactly what its real robot trajectory deserves.

What the gym does **not** show is that the interaction mesh beats keypoint matching once the same hard constraints are added. That ablation ties the reference at −0.14 [−0.56, +0.28]:
- The mesh keeps hand contacts much better.
- Keypoint matching keeps planted feet on the ground better.

The paper does not run this ablation either.

All numbers come from hidden-test runs:
- **Main run:** the combined run of job `9f74fa16-ef7`, in `redteam/hardening/results/combined_test/`.
- **Shipped methods:** bit-identical to the official run of the same job, in `redteam/hardening/results/official/` and `scoring/scores_test.json`.

**How the rank deltas are computed**
- Formula: weight × |coefficient| × (candidate − reference) / standard deviation, summed over all metrics and experiments, with each metric's sign taken from its better direction.
- Normalizer: the population standard deviation across the six honest methods of the run.
- Confidence intervals: paired bootstrap over test instances, where both methods are resampled on the same instances (`redteam/hardening/summarize.py`).
- The pre-hardening numbers are the red team's, computed on the old gym with its own normalizers. Before and after are therefore comparable in sign and ordering, not in absolute size.

## What changed

1. **The hidden test is truly hidden and larger.**
   - *Before:* train and test scored even and odd frames of the same two public 80-frame trajectories. The task JSON was editable, and the hidden evaluator trusted it as ground truth.
   - *Now:* `eval/test/hidden_tasks.py` is a self-contained copy of the demonstration generator with a private seed. At test time it creates 120 held-out instances in memory: 40 per experiment.
   - *Instance variety:* each instance has a new demonstrator size (0.90–1.15× the robot), new segment proportions, a new gait, a new box or ramp/column scene, and a new augmentation.
   - *Labels:* evaluation labels never touch disk. Public train data is 60 instances in `data/tasks/train/`, from `data/task_generator.py`.
   - *Independence:* the test reads nothing from `data/`, `scoring/`, or `method/`.
2. **Only the robot joint trajectory is scored.**
   - `eval/{train,test}/scoring_core.py` hold independent copies of the robot model and scene geometry. Every metric is recomputed from the submitted `q` with forward kinematics.
   - Ignored: any keypoints, metrics, flags, or score files a method writes.
   - Penetration is checked along every robot link (21 samples per link), not only at the 12 keypoints. Hands entering the object now count as penetration, as in the paper.
3. **A validity gate, with worst values on failure.** A method's whole run fails if it:
   - crashes, times out (2400 s per split), or misses any instance;
   - returns a malformed or wrong-length trajectory, or non-finite values;
   - leaves the joint limits, or exceeds the per-frame velocity limits (these were never checked before).

   A failed run gets the worst value of every metric in every experiment: duration 1, contact 0, and the cap for capped metrics (30 cm, 300 cm/s, 100 cm, 300 cm/s). Skipping an experiment no longer escapes its losses.
4. **Every method on disk is scored.**
   - A method is any `scripts/retargeters/<name>.sh <input_dir> <output_dir>`. Train and test discover all of them and run each through the same command.
   - The old whitelist read from editable `scores_train.json` is gone.
   - `evaluate_test.sh` takes no arguments.
5. **The paper's method is restored.**
   - *Solver:* a sequential convex solve of Eq. (3) per frame, as in App. D. Each iteration uses a Gauss-Newton model of the Laplacian residuals, linearised collision and stance constraints, exact joint and velocity bounds, and the trust region ‖dq‖ ≤ 0.2, for at most 10 iterations. It is warm-started from the previous frame.
   - *Stance:* the threshold is the paper's 1 cm/s, not 0.35 m/s. The old synthetic source had sliding "stance" feet; the new generator plants feet exactly.
   - *Removed:* the extra keypoint-matching term, the hard hand-contact equalities, and the retry that relaxed velocity bounds.
   - *Preprocessing:* direct scaling by α = h_robot/h_human (App. A, Eq. 7/8) is restored. The first job showed that skipping it (α = 1) made the robot float whenever the demonstrator was taller.
   - *Interaction mesh:* built from the scaled demonstration plus dense object and terrain samples, in the object frame. The baseline is per-frame keypoint IK on the directly scaled demonstration (GMR/PHC family).
6. **A new augmentation experiment** (`augmented_interaction_kinematics`, weight 0.30).
   - Source: paper Sec. III-B and App. C (Eq. 14).
   - The demonstration was recorded in one scene, but the robot must act in a modified one: a resized box with an exponentially decaying initial offset, or platform and column heights changed.
   - This is where preserving interaction matters. Copying keypoints misses the changed scene: the baseline's contact is 0.54 with 0.43 penetration duration.
7. **Contact covers planted feet as well as hands.** The definition is the paper's own duration measure with a 2 cm tolerance.
   - Planted feet are included in both nominal experiments. In the terrain task this matches the paper (hands, toes and heels).
   - In the object task, feet are added because otherwise a planar robot could float to reach the box at no cost.

## Do better results score higher? Variants, before → after

| Variant | Before | After | What the after-run shows |
|---|---:|---:|---|
| Genuinely better method: contact-aware OmniRetarget<br>(red-team "contact refinement" → `var_contact_aware`) | +1.109 | **+1.472** [+1.074, +1.898] | Adds demonstration-derived hand and foot contact costs to the mesh objective (App. B-5 allows task terms). Contact rises in all three experiments (0.83→0.96, 0.86→1.00, 0.71→0.94) with no extra penetration or skating. |
| Keypoint matching + contact costs + hard constraints (`var_keypoint_contact`) | — | **+1.659** [+1.258, +2.060] | Directly optimises what the metrics measure. An honest alternative, and headroom (see below). |
| No interaction mesh: keypoint objective, same constraints and solver (`var_no_mesh`) | **+0.347** | −0.138 [−0.557, +0.283] | No longer beats the reference, but tied within noise. Mesh: hand contact 0.99/0.97/0.96 vs 0.63/0.60/0.49. Feet: the ablation keeps them grounded better (0.98/0.96/0.91 vs 0.59/0.83/0.51). |
| Keypoint-IK baseline (`keypoint_ik`) | −3.577 | **−7.385** [−8.181, −6.567] | Penetration duration 0.38/0.36/0.43 and skating 0.41/0.54/0.56, vs ≈0 for the reference. Contact 0.71/0.86/0.54. |
| Constant standing pose (`sc_constant_pose`) | −9.562 | −3.336 [−3.952, −2.724] | Valid trajectory, so it is scored honestly: pelvis error 24–30 cm, contact 0.41/0.82/0.61. |
| More solver iterations | −0.448 | not rerun | The solver changed; the paper fixes 10 iterations. |
| Brute-force per-frame grid (3¹¹ configurations) | −5.675 | not rerun | The problem is continuous, and the paper's method does not exist to avoid enumeration. A coarse grid cannot meet the per-frame velocity limits while tracking the motion. |

Because methods are deterministic and the test is fixed, repeated runs are identical: seed noise is 0. The noise that remains comes from which instances are sampled. The paired bootstrap above measures it, and the better variant clears it by a wide margin. The test was enlarged from 12 to 40 instances per experiment to tighten these intervals.

## Shortcuts: fix and score before → after

All of these were run through `bash scripts/evaluate_test.sh` (no arguments), registered next to the shipped methods, in one combined hidden-test run.

| Shortcut | Before | Fix | After |
|---|---:|---|---:|
| Disconnected body (submitted keypoints not reachable by the robot) | **+3.116** | Keypoints are never read; geometry comes from `q` | Fake keypoints plus claimed metrics over the baseline's `q` scores exactly as the baseline: **−7.385**. Zero `q`: joint limit violated → failed, **−84.27** |
| Unchecked keypoints | **+2.250** | Same | Same as above |
| Mutable task (rewrite the public task files) | **+3.162** | Test instances are generated privately and never read from `data/` | Train inputs and annotations were overwritten with trivial content during the run. The shipped methods' test scores are bit-identical to the official run |
| Impossible `q` (1e200) | 0 (tie) | Validity gate | Failed (joint limit), **−84.27** |
| Empty submission | 0 (tie) | Missing output fails the run | **−84.27** |
| Partial: one experiment missing | 0 (tie) | Same | **−84.27** |
| Omit failed experiment (better object outputs, no terrain) | **+1.109** | Same | **−84.27** |
| Train-filter (drop candidate from `scores_train.json`) | 0 (tie) | Test no longer reads it; all registered methods scored | Train scores emptied during the run; no effect on test |
| Unlisted method name | 0 (tie) | No whitelist | Every new `sc_*` name was discovered and scored |
| NaN coordinates | undefined | Non-finite fails the run | **−84.27** |
| Malformed (0 frames) | no score (exit 1) | Shape check fails the run | **−84.27** |
| Score forgery (prewrite score files, claim metrics) | −2.79 | Scores always rewritten from outputs | Forged files overwritten; scored as its real trajectory: **−7.385** |
| Joint-limit or velocity-limit violation (new) | — | Gate | **−84.27** each |

## Shortcuts left open, and why

- **A method could read the hidden evaluator.** Methods run in the workspace that also holds `eval/test/`, so a method could import `eval/test/hidden_tasks.py` to recover labels, or the scorer to grade itself.
  - *Mitigation:* labels are never written to disk; `scoring/CONSTRAINTS.md` forbids reading or importing `eval/`; the separate grader checks for it.
  - *Why not closed:* closing it would mean moving or locking files while methods run. I judged that out of scope.
  - *Value of the leak:* small in practice. In nominal tasks the labels can already be inferred from the demonstration.
- **Optimising the measured quantities directly.** The metrics (penetration, skating, contact against the given scene, fidelity to the demonstration) can be computed from the inputs a method must receive anyway. `var_keypoint_contact` does roughly this and scores +1.66.
  - *Why this is not a loophole:* it outputs a valid robot motion. Its contacts come from the same demonstration information OmniRetarget uses, so it is an honest retargeting method (VideoMimic-style objectives, with hard constraints).
  - *What I did instead:* treated it as headroom. The per-split time limit of 2400 s is about 37× the reference's runtime, which bounds extra search.
  - *Why not closed:* closing it would require metrics the method cannot compute from its own input, and that would depart from the paper's metric definitions.
- **Copying the better variant.** The contact-aware variants live in `redteam/hardening/`, not in `method/`. A future agent who sees them could copy them; the separate grader judges novelty.

## Reference score before → after

The task changed, so the two columns are different problems.

**Before** (old public odd-frame slice, two trajectories):

| | penetration | skating dur. / max (cm/s) | contact | pelvis error | velocity error |
|---|---|---|---|---|---|
| object | 0 | 0.10 / 4.31 | 0.709 | 3.55 cm | 11.2 cm/s |
| terrain | 0 | 0 / 0 | 0.99996 | 20.3 cm | 72.2 cm/s |

On the old gym the reference also violated per-frame velocity limits (excess 0.63), which were not checked.

**After** (hidden test, 120 instances; `scoring/scores_test.json`):

| | penetration dur. / max (cm) | skating dur. / max (cm/s) | contact | pelvis error (cm) | velocity error (cm/s) |
|---|---|---|---|---|---|
| object | 0.006 / 0.07 | 0 / 0.37 | 0.827 | 1.04 | 4.45 |
| terrain | 0.013 / 0.20 | 0.002 / 0.72 | 0.856 | 0.99 | 9.07 |
| augmented | 0.005 / 0.16 | 0.001 / 0.67 | 0.714 | 1.45 | 7.20 |

Train (60 instances) is comparable: contact 0.835 / 0.897 / 0.830, penetration duration 0.019 / 0.005 / 0.048.

Against the keypoint-IK baseline (paper's Table II direction): penetration 0.38/0.36/0.43, skating 0.41/0.54/0.56, contact 0.71/0.86/0.54. So the reference wins on every constraint metric and on contact; it loses on pelvis and velocity fidelity (0.15–0.43 cm and 1.0–3.5 cm/s). The paper's numbers stay unchanged in `scoring/reference.json`, with a note that the planar setup is not on their cm scale. The augmented row there is this gym's measured value, because the paper reports no kinematic numbers for augmented motions.

## Headroom check

- **Where the reference sits at a bound:**
  - At or near: penetration duration (≤0.013), skating duration (≤0.002), maximum penetration (≤0.2 cm), maximum skating (≤0.72 cm/s).
  - Clearly below: contact (0.83 / 0.86 / 0.71), pelvis-path error (1.0–1.45 cm) and velocity error (4.5–9.1 cm/s).
- **Where the reference falls short:** its feet hover above the ground in many instances (foot contact 0.51–0.83). This is a real artifact of the paper's per-frame formulation in this setting: only stance sticking is enforced, and the mesh trades foot-ground for hand-object relations. It is also exactly what better methods fix.
- **What I added:** the augmentation experiment, where reference contact is lowest (0.71), and a test 3.3× larger.
- **Test 1, a better method beats the reference:** two variants exceed it well outside noise (+1.47, +1.66). Neither reaches the bound everywhere: velocity and pelvis errors stay well above zero, and the first variant's object contact is 0.96.
- **Test 2, simple methods stay below:**

  | Simple method | Rank vs reference |
  |---|---:|
  | Keypoint-IK baseline | −7.39 |
  | Constant pose | −3.34 |
  | No-mesh ablation | −0.14 (does not outrank) |

## Weights and coefficients

- **Experiment weights:**

  | Experiment | Weight |
  |---|---:|
  | object | 0.35 |
  | terrain | 0.35 |
  | augmented (new) | 0.30 |

  The nominal experiments are the paper's Table II; augmentation is the paper's second contribution.
- **Changes from the old coefficients:**
  - Contact preservation 0.8 → **1.0** (primary). The paper states that both contact preservation and penetration reduction are critical for downstream RL success.
  - Fidelity metrics: pelvis path 0.35 → **0.30**, velocity error 0.25 → **0.30**. Kept as guards against degenerate motions, but below the constraint and contact terms.
  - Constraint metrics unchanged: penetration duration 1.0, maximum penetration 0.5, skating duration 0.8, maximum skating 0.4.

## Could not fix / stated plainly

- **The mesh's own benefit is not established against an equally constrained keypoint method.** The two tie overall, and the gym shows the trade-off (hands versus feet) instead.
- **Scaled substitute, not the paper's setting.** It is a planar 11-DoF robot with sampled link collisions and procedurally generated demonstrations. There is no OMOMO, LAFAN1 or MoCap data, no 3D G1/H1/T1 robots, no PHC/GMR/VideoMimic implementations (one GMR/PHC-family baseline instead), and no downstream RL or hardware.
- **The subproblem solver differs.** It is solved exactly with SciPy SLSQP rather than an SOCP solver, plus an L∞ elastic slack on the linearised collision rows so the subproblem stays feasible.
  - *Not specified in the paper:* the frame-0 initialisation (keypoint IK on the scaled demonstration, then 40 iterations).
  - *My own choice:* demonstration scaling about the ground point below the pelvis.
- **Augmented contact differs between train and test** (0.83 vs 0.71), although train and test come from the same distribution. Twenty train instances per experiment remain a modest sample.
- **One unexplained instance in an earlier, smaller draw.** In the 36-instance draw of job `ca61137d-357`, one augmented-box instance had ~9.6 cm penetration in every frame for every method; I did not identify the cause. In the final 120-instance test every instance is solvable: some honest method reaches ≤0.4 cm penetration and ≥0.9 contact on each. The reference's worst penetration there is 1.73 cm, a forearm between its collision samples.
- **The variants were not re-scored on the final 60-instance train split.** The final job's combined train pass was cancelled because the node slowed about 60×. Variant train numbers exist only for the earlier 18-instance train draw (`redteam/hardening/results/combined_train_job2_18instances/`). The official train scores for the shipped methods are complete.
- **`validate.py` warns** that `eval/test/hidden_tasks.py` regenerates data. This is by design: the hidden generator creates test inputs, and scoring reads only the methods' outputs.
