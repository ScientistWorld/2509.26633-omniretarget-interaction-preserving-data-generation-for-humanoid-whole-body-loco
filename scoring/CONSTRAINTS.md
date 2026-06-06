# Constraints

Future methods must use the same task files, keypoint names, object geometry, terrain geometry, and evaluation tolerances when reporting comparable scores.

Do not improve scores by changing the benchmark generator, contact labels, terrain height function, object dimensions, or evaluator thresholds. A method may change the optimizer or retargeting algorithm, but it must output the same artifact schema under `checkpoints/retargeting/`.

Wall-clock runtime is not scored. Hardware-dependent timing should not be used as a claim.
