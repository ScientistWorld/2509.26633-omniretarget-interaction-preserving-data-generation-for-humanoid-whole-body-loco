# Direction

In scope: better ways to turn a human demonstration into a feasible robot
motion that keeps its interactions — e.g. better objectives or preprocessing for
morphology differences, better handling of contacts, clearances and planted
feet, better constraint handling and initialisation, whole-trajectory rather
than frame-by-frame optimisation, better transfer of interactions to modified
scenes.

Out of scope: anything that does not produce a valid robot joint trajectory for
every instance within the budget; reading evaluation labels or evaluator code;
changing the robot, scenes, generator, thresholds or evaluator; methods that
only optimise one side of the trade-off (e.g. matching keypoints while ignoring
penetration and skating, or satisfying contacts by abandoning the demonstrated
motion).
